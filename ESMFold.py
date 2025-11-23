import os
import requests

def get_protein_prediction_pdb(sequence):
    url = "https://biolm.ai/api/v3/esmfold/predict/"
    headers = {
        "Authorization": "Token 3307f4f01d4f574eb50542a8355484ed834452395975b86148e91e8e24c89924",
        "Content-Type": "application/json"
    }
    payload = {
        "items": [{"sequence": sequence}],
        "params": {
            "databases": ["mgnify", "small_bfd", "uniref90"],
            "predictions_per_model": 1,
            "relax": "none",
            "return_templates": True,
            "msa_iterations": 1,
            "max_msa_sequences": 1000,
            "algorithm": "mmseqs2"
        }
    }

    print(f"🔹 Sending sequence ({len(sequence)} aa) to BioLM...")

    response = requests.post(url, headers=headers, json=payload, timeout=180)
    print(f"🔹 Status: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            result = data["results"][0]

            # Get both the structure and the confidence scores
            pdb_str = result.get("pdb") or result.get("pdbs", [None])[0]
            plddt_values = result.get("plddt", [])

            # Save PDB
            pdb_file_path = os.path.join(os.getcwd(), 'prediction.pdb')
            with open(pdb_file_path, 'w') as f:
                f.write(pdb_str.strip() + "\nEND\n")

            # Save pLDDT if available
            if plddt_values:
                plddt_path = os.path.join(os.getcwd(), 'prediction_plddt.txt')
                with open(plddt_path, 'w') as f:
                    for i, score in enumerate(plddt_values, 1):
                        f.write(f"{i}\t{score}\n")
                print(f"✅ pLDDT scores saved at {plddt_path}")

            print(f"✅ PDB saved successfully at {pdb_file_path}")
            return pdb_file_path

        except Exception as e:
            print("❌ Failed to parse BioLM response:", e)
            print(response.text[:300])
            return None

    else:
        print(f"❌ Invalid response: Status {response.status_code}")
        print(response.text[:300])
        return None

def extract_plddt_average(pdb_file_path):
    total = 0
    count = 0
    with open(pdb_file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    bfactor = float(line[60:66].strip())  # B-factor column
                    total += bfactor
                    count += 1
                except:
                    continue
    if count == 0:
        return None
    return total / count


#get_protein_prediction_pdb("MSKIVRVGAVQSEPVWLDLEGSVDKTISLIEKAAADGVNVLGFPEVWIPGYPWSMWTSAVINNSHIIHDYMNNSMRKDSPQMKRIQAAVKEAGMVVVLGYSERDGASLYMAQSFIDPSGEIVHHRRKIKPTHI")





