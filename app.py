from flask import Flask, request, session, render_template
import requests, re
from pkg_resources import safe_listdir
import os


app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecret"

#model initialization


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, pipeline
import pandas as pd
import huggingface_hub

HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_hub.login(HF_TOKEN)
model_path = "Kinsleykinsley/tox21_predictor"
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", use_auth_token=True)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
df = pd.read_csv("hf://datasets/HUBioDataLab/tox21/data.csv")
label_cols = [col for col in df.columns if col != 'smiles']

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


EXCLUDED_COFACTOR_IDS = {
    "C00002", "C00003", "C00004", "C00005", "C00006",  # ATP, NADH, NAD+, NADPH, NADP+
    "C00016", "C00020", "C00024", "C00029", "C00035",  # CoA, AMP, UDP-Glucose, GDP, GTP
    "C00120", "C00122", "C00123", "C00138",            # FAD, FMN, SAM, Acetyl-CoA
    # add more as needed
}

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
        "reaction_id": rxn_key,
        "desired_reactant": id_to_name[desired_cid],
        "cofactors": cofactors,
        "products": products,
        "catalytic_cofactors": catalytic,
        "excluded_cofactors": excluded,
    }





@app.route('/')
@app.route('/about-page')
def about():
    return render_template('about-page.html')


@app.route('/home', methods = ["GET", "POST"])
def home():
    smiles_string = None
    compound_id = None
    rxn_result = None
    rxn_list = []
    reaction_ids = []
    products = []
    only_products = []
    product_smiles = []
    tox_scores = []
    tox = []
    if request.method == "POST":
        user_text = request.form.get("pollutant_text")
        if user_text:
            smiles_string = get_smiles_from_name(user_text)  # convert name → SMILES
            compound_id = get_kegg_compound_id(user_text)   # returns like "cpd:C00117"
            if compound_id:
                # strip "cpd:" → "C00117"

                compound_cid = compound_id.split(":")[1]

                rxns = get_kegg_reactions_for_compound(compound_id) or []  # ensure list

                if len(rxns) > 0:
                    rxn_result = f"Reactions involving {user_text} ({compound_id}):"
                    for rxn_id in rxns[:20]:
                        try:
                            equation = get_reaction_equation(rxn_id)
                            rxn_list.append(f"{rxn_id}: {equation}")
                            reaction_ids.append(f"{rxn_id[3:]}")  # drop "rn:"
                        except IndexError:
                            rxn_list.append(f"{rxn_id}: No equation available")

                    for rxn in reaction_ids:
                        result = get_products_from_reaction(rxn, compound_cid)
                        products.append(result)
                        only_products.append(result['products'])

                    for reaction in only_products:
                        reaction_smiles_products=[]
                        for product in reaction:
                            reaction_smiles_products.append(get_smiles_from_name(product))
                        product_smiles.append(reaction_smiles_products)

                    for reaction in product_smiles:
                        tox_scores.append(get_tox(reaction))

                    for score in tox_scores:
                        if "toxic" in score:
                            tox.append("Toxic")
                        if "unknown" in score:
                            tox.append("Toxicity could not be determined")
                        else:
                            tox.append("Safe")

                    for index, product in enumerate(products):
                        product["toxicity"] = tox[index]

                else:
                    rxn_result = "No reactions found for this compound."
            else:
                compound_id = "Compound not found in KEGG."

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
        tox_scores=tox_scores,
        tox=tox
    )


@app.route('/process-page')
def process():
    return render_template('process-page.html')

@app.route('/account-page')
def account():
    return render_template('account-page.html')

app.run(port=8000, debug=True)
