def encode_report(text,model_name="distilbert-base-uncased"):
    """Optional pretrained transformer encoder; first use may download model weights."""
    try: from transformers import AutoTokenizer,AutoModel
    except ImportError as e: raise RuntimeError("Install transformers to use this optional module") from e
    import torch
    tok=AutoTokenizer.from_pretrained(model_name); model=AutoModel.from_pretrained(model_name).eval()
    inputs=tok(text,return_tensors="pt",truncation=True,padding=True)
    with torch.no_grad(): return model(**inputs).last_hidden_state.mean(1).squeeze(0).numpy()
