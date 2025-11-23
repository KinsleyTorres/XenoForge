from flask import Flask, request, session, render_template, send_file
import requests, re
import json
#from pkg_resources import safe_listdir
import os
from generate_seq import (
    ImprovedHybridChemProteinModel,
    generate_sequences,
    analyze_sequence_quality,
    preprocess_smiles, predict_sequences,
    save_sequences_csv,
    save_sequences_fasta,
    generate_new_seqs)
from ESMFold import get_protein_prediction_pdb, extract_plddt_average
from prediction_models import get_temp, get_pH
from huggingface_hub import hf_hub_download
app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecret"

#model initialization


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, pipeline
import pandas as pd
import huggingface_hub
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from flask import render_template, request, redirect, url_for, flash

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            flash("Login successful!", "success")
            return redirect(url_for("account"))
        else:
            flash("Invalid username or password", "error")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose another.", "warning")
            return redirect(url_for("register"))

        # Create new user
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        # Automatically log the user in after registration
        login_user(new_user)
        flash("Account created successfully!", "success")
        return redirect(url_for("account"))

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


def make_prediction(file_path):
    df = pd.read_csv(file_path)
    df['chem_input'] = df['chem_input'].str.replace('<bos>', ' ')
    df['chem_input'] = df['chem_input'].str.replace('<sep>', ' ')
    df['chem_input'] = df['chem_input'].str.replace('>', '</s>')
    df.to_csv('ester_hydrolase_sequences.csv', index=False)


import os
HF_TOKEN = os.getenv("HF_TOKEN")
huggingface_hub.login(HF_TOKEN)
model_path = "Kinsleykinsley/tox21_predictor"
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", use_auth_token=True)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
df = pd.read_csv("hf://datasets/HUBioDataLab/tox21/data.csv")
label_cols = [col for col in df.columns if col != 'smiles']

model_path2 = "Kinsleykinsley/Location_Molformer"
model2 = AutoModelForSequenceClassification.from_pretrained(model_path2, trust_remote_code=True)
tokenizer2 = AutoTokenizer.from_pretrained(model_path2, trust_remote_code=True)
classifier2 = pipeline("text-classification", model=model2, tokenizer=tokenizer2, top_k=None)


model_path3 = "Kinsleykinsley/EC_Molformer"
model3 = AutoModelForSequenceClassification.from_pretrained(model_path3, trust_remote_code=True)
tokenizer3 = AutoTokenizer.from_pretrained(model_path3, trust_remote_code=True)
classifier3 = pipeline("text-classification", model=model3, tokenizer=tokenizer3)


def get_tox(metabolites):
    tox_ranking = []
    for text in metabolites:
        if not text or not isinstance(text, str):
            tox_ranking.append("unknown")
            continue

        results = classifier(text)
        scores = results[0]

        score_dict = {}
        for score_info in scores:
            label_index = int(score_info['label'].split('_')[1])
            label_name = label_cols[label_index]
            score_dict[label_name] = score_info['score']

        if all(val < 0.2 for val in score_dict.values()):
            tox_ranking.append("safe")
        else:
            tox_ranking.append("toxic")

    return tox_ranking

def get_location(text):
    result = classifier2(text)
    print(result)
    return result

def get_ec_class(text):
    result = classifier3(text)
    print(result)
    return result


EXCLUDED_COFACTOR_IDS = {
    "C00002", "C00003", "C00004", "C00005", "C00006",  # ATP, NADH, NAD+, NADPH, NADP+
    "C00016", "C00020", "C00024", "C00029", "C00035",  # CoA, AMP, UDP-Glucose, GDP, GTP
    "C00120", "C00122", "C00123", "C00138",            # FAD, FMN, SAM, Acetyl-CoA
    # add more as needed
}

EXCLUDED_COFACTORS = ["ATP", "NADH", "NAD+", "NADPH", "NADP+"
    "CoA", "AMP", "UDP-Glucose", "GDP", "GTP"
    "FAD", "FMN", "SAM", "Acetyl-CoA"]

CATALYTIC_COFACTOR_IDS = {
    "C00001",  # H2O
    "C00007",  # Oxygen
    "C00027",  # Hydrogen peroxide
    "C00080",  # H+
    "C00009",  # Phosphate
    "C00255",  # Thiamine diphosphate (TPP)
    "C00018",  # Pyridoxal phosphate (PLP)
}



#step 1 fn
def get_smiles_from_name(name):
    url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return None

#step 2 fns
def get_kegg_compound_id(name):
    res = requests.get(f"http://rest.kegg.jp/find/compound/{name}").text.strip()
    lines = res.split("\n")
    if not lines or lines == ['']:
        return None
    parts = lines[0].split("\t")
    return parts[0] if parts else None


# Given a KEGG compound ID, find all linked reaction IDs
def get_kegg_reactions_for_compound(cid):
    res = requests.get(f"http://rest.kegg.jp/link/rn/{cid}").text.strip()
    rxn_ids = []
    for line in res.split("\n"):
        parts = line.split("\t")
        if len(parts) > 1:   # make sure there's a reaction ID
            rxn_ids.append(parts[1].strip())
    return rxn_ids


# Get the reaction equation from a reaction ID (e.g., rn:R00622)
def get_reaction_equation(rxn_id):
    res = requests.get(f"http://rest.kegg.jp/get/{rxn_id}").text
    for line in res.splitlines():
        if line.startswith("EQUATION"):
            equation = line.replace("EQUATION", "").strip()
            return equation


def _norm(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.split(";")[0].strip().lower()
    # remove common stereochemical prefixes
    s = re.sub(r'^(alpha-|beta-|d-|l\-)', '', s)
    s = s.replace("’", "'").replace("\u2013", "-")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# reuse your existing get_compound_name if you want; but this helper uses it
def get_compound_name(cid):
    url = f"http://rest.kegg.jp/get/cpd:{cid}"
    res = requests.get(url).text
    for line in res.splitlines():
        if line.startswith("NAME"):
            return line.replace("NAME", "").strip().split(";")[0]
    return cid

def get_products_from_reaction(rxn_id, desired_cid):
    """
    Parse KEGG reaction and return desired reactant, cofactors, and products.
    Uses KEGG compound IDs for matching instead of names.
    """

    # ensure rn: prefix
    rxn_key = rxn_id if rxn_id.lower().startswith("rn:") else f"rn:{rxn_id}"

    # fetch KEGG entry
    url = f"http://rest.kegg.jp/get/{rxn_key}"
    res = requests.get(url).text

    # grab equation
    equation_line = None
    for line in res.splitlines():
        if line.startswith("EQUATION"):
            equation_line = line.replace("EQUATION", "").strip()
            break
    if not equation_line:
        return None

    # split sides
    if "<=>" in equation_line:
        lhs, rhs = equation_line.split("<=>", 1)
    elif "=>" in equation_line:
        lhs, rhs = equation_line.split("=>", 1)
    else:
        return None

    # extract compound IDs
    left_ids = re.findall(r'C\d{5}', lhs)
    right_ids = re.findall(r'C\d{5}', rhs)
    all_ids = left_ids + right_ids
    if not all_ids:
        return None

    # map IDs → names (for readability)
    id_to_name = {cid: get_compound_name(cid) for cid in all_ids}
    left_names = [id_to_name[cid] for cid in left_ids]
    right_names = [id_to_name[cid] for cid in right_ids]

    # make sure desired_cid is in the reaction
    if desired_cid not in all_ids:
        return None

    # assign cofactors/products depending on which side the desired reactant is on
    if desired_cid in left_ids:
        cofactors = [id_to_name[cid] for cid in left_ids if cid != desired_cid]
        products = [id_to_name[cid] for cid in right_ids]
    elif desired_cid in right_ids:
        cofactors = [id_to_name[cid] for cid in right_ids if cid != desired_cid]
        products = [id_to_name[cid] for cid in left_ids]
    else:
        return None  # should never happen

    catalytic = [id_to_name[cid] for cid in all_ids if cid in CATALYTIC_COFACTOR_IDS]
    excluded = [id_to_name[cid] for cid in all_ids if cid in EXCLUDED_COFACTOR_IDS]

    return {
        "reaction id": rxn_key,
        "reaction equation": equation_line,
        "desired reactant": id_to_name[desired_cid],
        "substrates": cofactors,
        "products": products
        #"catalytic_substrates": catalytic,
        #"excluded_cofactors": excluded,
    }



amide_file_path = hf_hub_download(repo_id="Kinsleykinsley/nitrite_model", filename="pytorch_eos_progen2-small_epoch_6.pt")
ester_file_path = hf_hub_download(repo_id="Kinsleykinsley/ester_model", filename="pytorch_eos_progen2-small_epoch_12.pt")
glycosyl_file_path = hf_hub_download(repo_id="Kinsleykinsley/glycosyl_model", filename="pytorch_eos_progen2-small_epoch_15.pt")
halide_file_path = hf_hub_download(repo_id="Kinsleykinsley/halide_model", filename="pytorch_eos_progen2-small_epoch_8.pt")

@app.route('/')
@app.route('/about-page')
def about():
    return render_template('about-page.html')


@app.route('/home', methods=["GET", "POST"])
def home():
    smiles_string = None
    compound_id = None
    rxn_result = None
    rxn_list = []
    reaction_ids = []
    products = []
    only_products = []
    only_substrates = []
    product_smiles = []
    substrate_smiles = []
    combined_smiles = []
    formatted_reactions = []
    tox_scores = []
    tox = []
    location = []
    reaction_validity = []
    aminoacids = []
    selected_reaction = None
    selected_reaction_details = None
    pollutant_name = None

    if request.method == "POST":
        print("=" * 50)
        print("POST REQUEST RECEIVED")
        print("Form data:", request.form)
        print("=" * 50)

        user_text = request.form.get("pollutant_text")
        selected_reaction = request.form.get("reaction")
        hidden_pollutant = request.form.get("hidden_pollutant_text")

        print(f"user_text: '{user_text}'")
        print(f"selected_reaction: '{selected_reaction}'")
        print(f"hidden_pollutant: '{hidden_pollutant}'")

        # If a reaction is selected (not empty string), get the original pollutant name from hidden field
        if selected_reaction and selected_reaction.strip():  # Check for non-empty string
            print("A reaction was selected!")
            user_text = hidden_pollutant
            print(f"Using hidden pollutant text: {user_text}")

        if user_text:
            pollutant_name = user_text
            print(f"Processing pollutant: {pollutant_name}")
            smiles_string = get_smiles_from_name(user_text)
            compound_id = get_kegg_compound_id(user_text)

            if compound_id:
                compound_cid = compound_id.split(":")[1]
                rxns = get_kegg_reactions_for_compound(compound_id) or []

                if len(rxns) > 0:
                    rxn_result = f"Reactions involving {user_text} ({compound_id}):"

                    # --- Fetch reaction equations ---
                    for rxn_id in rxns[:40]:
                        try:
                            equation = get_reaction_equation(rxn_id)
                            rxn_list.append(f"{rxn_id}: {equation}")
                            reaction_ids.append(f"{rxn_id[3:]}")
                        except IndexError:
                            rxn_list.append(f"{rxn_id}: No equation available")

                    # --- Get substrates and products ---
                    for idx, rxn in enumerate(reaction_ids):
                        result = get_products_from_reaction(rxn, compound_cid)
                        result["reaction_id"] = f"rn:{rxn}"  # ADD REACTION_ID HERE!!!

                        result_list = []
                        for item in result['substrates']:
                            if item in EXCLUDED_COFACTORS:
                                result_list.append(item)
                        if 'H2O' in result['substrates']:
                            if len(result['substrates']) == 1:
                                if len(result_list) == 0:
                                    products.append(result)
                                    only_products.append(result['products'])
                                    only_substrates.append(result['substrates'])
                                    valid = 'applicable'
                                    reaction_validity.append(valid)
                        else:
                            valid = "non applicable"
                            reaction_validity.append(valid)
                            products.append(result)
                            only_products.append(result['products'])
                            only_substrates.append(result['substrates'])

                    for idx, (substrates, products_list) in enumerate(zip(only_substrates, only_products)):
                        substrate_smiles_list = []
                        product_smiles_list = []

                        # Add user compound as substrate if missing
                        if user_text not in substrates:
                            substrates = [user_text] + substrates

                        # substrates
                        for substrate in substrates:
                            s = get_smiles_from_name(substrate)
                            if s:
                                substrate_smiles_list.append(s)
                            else:
                                print(f"no smiles available for substrate: {substrate}")

                        # products
                        for product in products_list:
                            p = get_smiles_from_name(product)
                            if p:
                                product_smiles_list.append(p)
                            else:
                                print(f"no smiles available for product: {product}")

                        substrate_smiles.append(substrate_smiles_list)
                        product_smiles.append(product_smiles_list)

                        # --- Format combined SMILES string ---
                        if substrate_smiles_list or product_smiles_list:
                            formatted = f"<bos> {'.'.join(substrate_smiles_list)} >> {'.'.join(product_smiles_list)} <sep>"


                            # --- Get model output and map to binary label ---
                            raw_ec = get_ec_class(formatted)

                            class_list = {"LABEL_0": "ester hydrolase", "LABEL_1": "glycosyl hydrolase",
                                          "LABEL_2": "amide/nitrile hydrolase", "LABEL_3": "halide hydrolase"}

                            products[idx]["formatted_sequence"] = formatted
                            products[idx]["ec"] = f"{raw_ec[0]['label']}"
                            products[idx]["ec_class"] = ""
                            products[idx]["confidence"] = f"{raw_ec[0]['score']}"
                            if products[idx]["ec"]:
                                for key, value in class_list.items():
                                    if products[idx]["ec"] == key:
                                        products[idx]["ec_class"] = value
                        else:
                            print("no smiles available for this reaction")

                    # --- Example toxicity scoring ---
                    for reaction in product_smiles:
                        tox_scores.append(get_tox(reaction))

                    for score in tox_scores:
                        if "toxic" in score:
                            tox.append("Toxic")
                        elif "unknown" in score:
                            tox.append("Toxicity could not be determined")
                        else:
                            tox.append("Safe")

                    for index, product in enumerate(products):
                        product["toxicity"] = tox[index] if index < len(tox) else "Unknown"
                        product["validity"] = reaction_validity[index] if index < len(reaction_validity) else "Unknown"

                    print(f"\nTotal products: {len(products)}")

                    # Print all reaction IDs for debugging
                    print("\nAll reaction IDs in products:")
                    for i, product in enumerate(products):
                        print(f"  Product {i}: reaction_id = '{product.get('reaction_id')}'")

                    # --- Find the selected reaction details ---
                    if selected_reaction and selected_reaction.strip():
                        print(f"\n*** Looking for selected reaction: '{selected_reaction}' ***")
                        for i, product in enumerate(products):
                            reaction_id = product.get('reaction_id')
                            if reaction_id == selected_reaction:
                                selected_reaction_details = product
                                print(f"*** MATCH FOUND at index {i}! ***")
                                break

                        if not selected_reaction_details:
                            print(f"*** NO MATCH FOUND for '{selected_reaction}' ***")
                    else:
                        print("\nNo reaction selected (or empty)")

                else:
                    rxn_result = "No reactions found for this compound."
            else:
                compound_id = "Compound not found in KEGG."
        else:
            print("No user_text provided!")


    model_list= {"ester hydrolase": ester_file_path, "glycosyl hydrolase": glycosyl_file_path, "amide/nitrile hydrolase": amide_file_path, "halide hydrolase": halide_file_path}
    print(f"\nFinal values before render:")
    print(f"selected_reaction: '{selected_reaction}'")
    print(f"selected_reaction_details: {selected_reaction_details}")
    print(f"pollutant_name: {pollutant_name}")
    if selected_reaction:
        for key, value in model_list.items():
            if selected_reaction_details["ec_class"] == key:
                predicted_model_path = value
                original_input = selected_reaction_details["formatted_sequence"]
                original_input.replace('<bos>', ' ')
                original_input.replace('<sep>', ' ')
                formatted_seq = original_input.replace('>', '</s>')
                generate_new_seqs(predicted_model_path, formatted_seq)
                return redirect(url_for('sequence_landing_page'))

    return render_template(
        "home.html",
        user_text=smiles_string,
        compound_id=compound_id,
        rxn_result=rxn_result,
        rxn_list=rxn_list,
        reaction_ids=reaction_ids,
        products=products,
        only_products=only_products,
        product_smiles=product_smiles,
        substrate_smiles=substrate_smiles,
        tox_scores=tox_scores,
        tox=tox,
        reaction_validity=reaction_validity,
        selected_reaction=selected_reaction,
        selected_reaction_details=selected_reaction_details,
        pollutant_name=pollutant_name
    )


@app.route('/sequence_landing_page')
def sequence_landing_page():
    seq_df = pd.read_csv(r'C:\Users\admin\PycharmProjects\XenoForge\predictions\example1_sequences.csv')
    data_for_template = seq_df.to_dict(orient='records')
    return render_template('sequence_landing_page.html', rows=data_for_template)



@app.route('/view_in_3d', methods=['GET', 'POST'])
def view_in_3d():
    requested_sequence = None
    pdb_file_url = None
    avg_plddt = None
    pH = None
    temp = None

    if request.method == "POST":
        requested_sequence = request.form.get("selected_row")
        if requested_sequence:
            pdb_file_path = get_protein_prediction_pdb(requested_sequence)



            if pdb_file_path:
                # Serve the actual PDB via Flask route
                pdb_file_url = url_for('serve_pdb_file')
                avg_plddt = extract_plddt_average(pdb_file_path)
                temp = get_temp(requested_sequence)
                pH = get_pH(requested_sequence)
                return render_template('view_in_3d.html',
                                       requested_sequence=requested_sequence,
                                       pdb_file=pdb_file_url,
                                       avg_plddt=avg_plddt, temp=temp, pH=pH)
            else:
                return "Error generating PDB. Check server logs.", 500

    requested_sequence = request.args.get('seq')
    if requested_sequence:
        pdb_file_url = url_for('serve_pdb_file')

    return render_template('view_in_3d.html',
                           requested_sequence=requested_sequence,
                           pdb_file=pdb_file_url)


@app.route('/serve_pdb_file')
def serve_pdb_file():
    pdb_path = os.path.join(os.getcwd(), 'prediction.pdb')
    if os.path.exists(pdb_path):
        return send_file(pdb_path, mimetype='chemical/x-pdb')
    return "PDB file not found", 404

@app.route('/process-page')
def process():
    return render_template('process-page.html')

@app.route("/account-page")
def account():
    if current_user.is_authenticated:
        return render_template("profile-page.html")
    return render_template("account-page.html")

app.run(port=8000, debug=True)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

