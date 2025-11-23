from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def get_temp(seq):
    model_path = "Kinsleykinsley/temp_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, subfolder = 'temp_model')
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    inputs = tokenizer(
        seq,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_norm = outputs.logits.squeeze().item()
    print("Normalized output:", pred_norm)

    mean = 49.62447245017585

    std = 18.42132419793425

    # convert to real temperature
    real_temp = pred_norm * std + mean
    print("Predicted optimal temperature:", real_temp)

    return real_temp

#get_temp("MLVRSFLGFAVLAATCLAASLQEVTEFGDNPTNIQMYIYVPDQLDTNPPVIVALHPCGGSAQQWFSGTQLPSYADDNGFILIYPSTPHMSNCWDIQNPDTLTHGQGGDALGIVSMVNYTLDKHSGDSSRVYAMGFSSGGMMTNQLAGSYPDVFEAGAVYSGVAFGCAAGAESATPFSPNQTCAQGLQKTAQEWGDFVRNAYAGYTGRRPRMQIFHGLEDTLVRPQCAEEALKQWSNVLGVELTQEVSGVPSPGWTQKIYGDGTQLQGFFGQGIGHQSTVNEQQLLQWFGLI")


from transformers import pipeline

def get_pH(seq):
    model_path = "Kinsleykinsley/pH_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, subfolder = 'pH_model')
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    predictor = pipeline("text-classification", model=model, tokenizer=tokenizer)

    result = predictor(seq)
    pH = result[0]['score']
    print(result)
    print(pH)
    return pH

#get_pH("MRNGLLKVAALAAASAVNGENLAYSPPFYPSPWANGQGDWAEAYQKAVQFVSQLTLAEKVNLTTGTGWEQDRCVGQVGSIPRLGFPGLCMQDSPLGVRDTDYNSAFPAGVNVAATWDRNLAYRRGVAMGEEHRGKGVDVQLGPVAGPLGRSPDAGRNWEGFAPDPVLTGNMMASTIQGIQDAGVIACAKHFILYEQEHFRQGAQDGYDISDSISANADDKTMHELYLWPFADAVRAGVGSVMCSYNQVNNSYACSNSYTMNKLLKSELGFQGFVMTDWGGHHSGVGSALAGLDMSMPGDIAFDSGTSFWGTNLTVAVLNGSIPEWRVDDMAVRIMSAYYKVGRDRYSVPINFDSWTLDTYGPEHYAVGQGQTKINEHVDVRGNHAEIIHEIGAASAVLLKNKGGLPLTGTERFVGVFGKDAGSNPWGVNGCSDRGCDNGTLAMGWGSGTANFPYLVTPEQAIQREVLSRNGTFTGITDNGALAEMAAAASQADTCLVFANADSGEGYITVDGNEGDRKNLTLWQGADQVIHNVSANCNNTVVVLHTVGPVLIDDWYDHPNVTAILWAGLPGQESGNSLVDVLYGRVNPGKTPFTWGRARDDYGAPLIVKPNNGKGAPQQDFTEGIFIDYRRFDKYNITPIYEFGFGLSYTTFEFSQLNVQPINAPPYTPASGFTKAAQSFGQPSNASDNLYPSDIERVPLYIYPWLNSTDLKASANDPDYGLPTEKYVPPNATNGDPQPIDPAGGAPGGNPSLYEPVARVTTIITNTGKVTGDEVPQLYVSLGGPDDAPKVLRGFDRITLAPGQQYLWTTTLTRRDISNWDPVTQNWVVTNYTKTIYVGNSSRNLPLQAPLKPYPGI")
