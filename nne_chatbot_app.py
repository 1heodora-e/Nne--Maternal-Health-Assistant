"""
Nne – Maternal Health Chatbot (Gradio)
Use for: (1) Demo video – run locally with GPU; set GRADIO_SHARE=1 for a public link.
         (2) Hugging Face Space – upload this file + requirements.txt (as requirements.txt)
             and your adapter (or set ADAPTER_PATH to a Hub repo); runs on CPU (slower).

Install: pip install -r requirements.txt
Run:     python app.py
         For shareable link (demo): set GRADIO_SHARE=1 then python app.py
"""

import json
import os
import re
import tempfile
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr

# -----------------------------------------------------------------------------
# Config – override with env vars if needed
# -----------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2-2b-it")
# Adapter: local path (e.g. outputs_nne/nne_exp2_high_lr) or Hub repo (e.g. username/nne-exp2)
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "outputs_nne/nne_exp2_high_lr")
# 64 tokens keeps responses concise and inference fast on CPU (~40-50 s); increase for longer answers on GPU
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
# Set GRADIO_SHARE=1 for a public link when running locally (e.g. for demo video)
SHARE = os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")

# Hugging Face token (required for gated Gemma)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get(
    "HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
)

INSTRUCTION = (
    "You are Nne, a wise and empathetic midwife assistant. "
    "You ONLY answer questions about pregnancy, childbirth, postpartum, breastfeeding, and maternal or newborn health. "
    "Assess the situation, then answer with care and clinical awareness. "
    "If the question is not about these topics, politely say that you are here only for maternal health questions "
    "and invite the user to ask something about pregnancy, postpartum, or maternal wellness."
)

# Shown when we detect off-topic input (no maternal keyword; no model call)
OFF_TOPIC_REPLY = (
    "I'm Nne, a maternal health chatbot, and I only respond to questions about pregnancy, postpartum, newborn, "
    "reproductive, and women's health. What women's health related thing would you like to learn?"
)

# Guardrail: only call model when query has a maternal keyword (from nne_mixed_train_v2.jsonl scope). Otherwise redirect.
MATERNAL_KEYWORDS = (
    # Pregnancy & gestation
    "pregnancy", "pregnant", "gestation", "gestational", "trimester", "prenatal", "antenatal",
    "fetal", "fetus", "foetal", "foetus", "uterus", "uterine", "obstetric", "obstetrics",
    "childbirth", "child birth", "due date", "baby bump", "conception", "conceive", "implantation",
    # Labour, delivery, birth
    "labor", "labour", "delivery", "birth", "contraction", "cesarean", "caesarean", "c-section",
    "vbac", "breech", "placenta", "placental", "placenta previa", "placenta accreta", "placental abruption",
    "oxytocin", "uterine rupture", "uterine atony", "uterine inversion", "vaginal birth", "vaginal delivery",
    # Postpartum & maternal
    "postpartum", "post-partum", "post partum", "postnatal", "post-natal", "maternal", "mother",
    "mom", "mum", "new mother", "postpartum depression", "postpartum hemorrhage", "postpartum bleeding",
    "vaginal hematoma", "lochia",
    # Newborn & infant
    "newborn", "newborns", "neonate", "neonatal", "infant", "infants", "baby", "babies",
    "preterm", "prematurity", "premature birth", "nicu", "neonatal respiratory", "neonatal distress",
    "after birth", "at birth", "months after birth",
    # Breastfeeding & lactation
    "breastfeed", "breastfeeding", "breast feed", "lactation", "lactating", "nursing", "wean", "weaning",
    "prolactin", "mammary", "milk production", "breast milk",
    # Reproductive anatomy & conditions
    "vagina", "vaginal", "vulva", "vulvar", "cervix", "cervical", "ovary", "ovarian", "fallopian",
    "endometrium", "endometriosis", "leiomyoma", "fibroid", "fibroids", "ectopic pregnancy", "ectopic",
    # Pregnancy loss
    "miscarriage", "stillbirth", "still birth", "spontaneous abortion", "pregnancy loss", "recurrent miscarriage",
    # Pregnancy tests & markers
    "afp", "alpha-fetoprotein", "alpha fetoprotein", "hcg", "beta hcg", "pregnancy test", "amniotic",
    "amniocentesis", "nuchal", "neural tube",
    # Pregnancy complications
    "preeclampsia", "pre-eclampsia", "eclampsia", "gestational diabetes", "morning sickness", "hyperemesis",
    "obstetric complication", "obstetric complications", "hemorrhage", "haemorrhage", "pprom",
    "rupture of membranes", "premature rupture",
    # Fertility & contraception
    "fertility", "infertility", "infertile", "ovulation", "ovulate", "contraception", "contraceptive",
    "emergency contraceptive", "iud", "oral contraceptive", "ocp", "ovarian reserve", "follicle", "follicular",
    # Hormones & menopause
    "estrogen", "oestrogen", "progesterone", "folate", "folic acid", "menopause", "menopausal",
    "hot flash", "hot flashes", "hormone replacement", "hrt", "night sweats", "vaginal dryness",
    # Female / women's health general
    "female", "woman", "women", "gynecolog", "gynaecolog", "midwife", "midwifery",
    # Menstrual & pelvic
    "menstrual", "menstruation", "period", "menorrhagia", "dysmenorrhea", "amenorrhea", "pelvic",
    "pelvic pain", "vaginal bleeding", "vaginal discharge", "heavy bleeding", "irregular period",
    # STI & sexual health
    "sexually transmitted", "sti", "chlamydia", "gonorrhea", "syphilis", "trichomonas", "hpv",
    "bacterial vaginosis", "bv", "pelvic inflammatory", "pid", "postcoital", "chancre",
    # Gynecologic conditions & cancer
    "cervical cancer", "ovarian cancer", "uterine cancer", "endometrial", "endometrial cancer",
    "gestational trophoblastic", "trophoblastic", "molar pregnancy", "molar", "ectopic",
    "lichen sclerosis", "lichen sclerosus", "vulvar lichen", "thecoma", "granulosa cell",
    "prolactinoma", "sex cord stromal",
    # Breast (women's health)
    "breast", "mastitis", "periductal mastitis", "intraductal papilloma", "breast cancer",
    "mammogram", "breast lump", "nipple discharge", "areola",
    # Screening & exams
    "pap smear", "pap test", "cervical screening", "pelvic exam", "breast exam",
    # Other women's health from dataset
    "septate uterus", "uterine anomaly", "tamoxifen", "celiac disease", "folate deficiency",
    "nne", "maternal health", "maternal wellness", "pregnancy care", "prenatal care", "antenatal care",
    "postnatal care", "newborn care", "baby care", "rh alloimmunization", "rh negative", "rh positive",
    "birth defect", "congenital infection", "neonatal sepsis", "galactosemia", "entamoeba", "amebiasis",
)


def _is_off_topic(text: str) -> bool:
    """Return True if the message is not clearly about maternal health (no maternal keyword). Only call model when on-topic."""
    t = text.lower().strip()
    if not t:
        return True
    has_maternal = any(k in t for k in MATERNAL_KEYWORDS)
    return not has_maternal


def load_model_and_tokenizer():
    """Load base model (float16) + PEFT adapter and tokenizer. Works on GPU or CPU."""
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH if os.path.isdir(ADAPTER_PATH) else MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    # Works for both local paths and Hub repo IDs (e.g. "username/nne-adapter")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    instruction: str,
    user_input: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    prompt = f"{instruction}\n\nInput: {user_input}\n\nOutput:"
    # 512 matches the max_target_length used during training; prevents OOM on long inputs
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return text.strip()


def _strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> block (for model only; not shown to user)."""
    if not text:
        return text
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def nne_reply(message, history):
    """Gradio chat handler: user message -> model reply. Domain-limited to maternal health."""
    if not message or not message.strip():
        return ""
    msg = message.strip()
    if _is_off_topic(msg):
        return OFF_TOPIC_REPLY
    reply = generate_answer(model, tokenizer, INSTRUCTION, msg)
    return _strip_thinking(reply)


# -----------------------------------------------------------------------------
# Load model once at startup
# -----------------------------------------------------------------------------
print("Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer()
print("Model ready.")


# -----------------------------------------------------------------------------
# Pink, girly, calm theme + custom CSS
# -----------------------------------------------------------------------------
PINK_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.rose,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="linear-gradient(180deg, #9D174D 0%, #831843 50%, #701A75 100%)",
    block_background_fill="#BE185D",
    block_border_color="#F8B4C4",
    block_label_background_fill="#FCE4EC",
    block_title_text_color="#9D174D",
    button_primary_background_fill="linear-gradient(90deg, #EC4899 0%, #DB2777 100%)",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#BE185D",
    input_background_fill="#FFFBFB",
    input_border_color="#F9A8D4",
)

CUSTOM_CSS = """
/* Rounded, soft panels and chat bubbles */
.gradio-container { font-family: 'Segoe UI', 'Georgia', sans-serif !important; }
.block { border-radius: 16px !important; box-shadow: 0 4px 20px rgba(236, 72, 153, 0.08) !important; }
.gr-button { border-radius: 12px !important; font-weight: 600 !important; }
.gr-box { border-radius: 12px !important; }
/* Sidebar styling - dark pink frames with padding from edges */
#sidebar { padding: 16px 20px !important; }
#sidebar .block { background: linear-gradient(180deg, #BE185D 0%, #9D174D 100%) !important; border: 1px solid #831843 !important; padding: 16px !important; margin-bottom: 12px !important; }
#sidebar .block:last-child { margin-bottom: 0 !important; }
#sidebar .gr-markdown { color: #831843 !important; }
/* Chat area - dark pink message bubbles */
.message.user, .message.bot { color: #1f2937 !important; }
.message.user { background: linear-gradient(135deg, #C73659 0%, #9D174D 100%) !important; border-radius: 18px 18px 4px 18px !important; }
.message.bot { background: linear-gradient(135deg, #BE185D 0%, #831843 100%) !important; border-radius: 18px 18px 18px 4px !important; border: 1px solid #831843 !important; }
/* Error/alert text - black so visible on light backgrounds */
.alert, .error, .gr-alert, [class*="err"] { color: #111 !important; }
"""

# -----------------------------------------------------------------------------
# UI copy – tell what the chatbot does, intuitive instructions
# -----------------------------------------------------------------------------
ABOUT_MARKDOWN = """
### 🌸 About Nne

**Nne** (Igbo for *Mother*) is an AI assistant focused on **maternal health**.

I provide supportive, evidence-informed information on:
- **Pregnancy** – from early weeks to delivery  
- **Postpartum** – recovery, feeding, mood  
- **General maternal wellness** – when to seek help and how to care for yourself  

*I am here to inform and support, not to replace your healthcare provider. For emergencies, always contact a clinician or emergency services.*
"""

HOW_TO_USE_MARKDOWN = """
### How to Use

1. **Type** your question in the input box below (e.g. morning sickness, breastfeeding, when to see a doctor).
2. **Press Enter** or click **Ask Nne** to get a reply.
3. Use **Clear Chat** to start a new conversation.
4. Use **Export chat** to download your conversation as a JSON file.
5. **Tip:** The more specific your question, the more focused the answer. I only answer questions about maternal health.
"""

TAGLINE = "Empowering mothers with calm, caring support. How can I help you today?"

PLACEHOLDER = "Ask me about pregnancy, postpartum, breastfeeding, or maternal health..."


def chat_submit(message, history):
    """On submit: append user message, get reply, append assistant message (Gradio 6 role/content format)."""
    if not message or not message.strip():
        return history, "", history
    reply = nne_reply(message, history)
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history, "", new_history


def chat_clear():
    """Clear chat history."""
    return [], ""


def init_chat():
    """Show welcome message on load (Gradio 6 role/content format)."""
    welcome = (
        "Hello! I'm **Nne**, your maternal health companion. I'm here to offer supportive, "
        "evidence-informed information on pregnancy, postpartum, and maternal wellness. What would you like to ask today?"
    )
    initial = [{"role": "assistant", "content": welcome}]
    return initial, initial


def export_chat(history):
    """Export chat history to a JSON file and return path for download (history is list of {role, content} dicts)."""
    if not history:
        return None
    records = [
        {"role": m["role"], "content": m.get("content", "") or ""} for m in history
    ]
    name = f"nne_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # Write under cwd so Gradio can serve the file reliably (e.g. on Hugging Face Spaces)
    path = os.path.join(os.getcwd(), name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Build UI: sidebar (About + How to Use) + main (title, chat, input, buttons)
# -----------------------------------------------------------------------------
with gr.Blocks(title="Nne – Maternal Health Assistant") as demo:
    gr.HTML(
        '<div style="text-align: center; padding: 12px 0; border-bottom: 2px solid #F9A8D4; margin-bottom: 16px;">'
        '<span style="font-size: 1.5rem; font-weight: 700; color: #9D174D;">🌸 Nne – Maternal Health Assistant</span>'
        "</div>"
    )
    gr.Markdown(
        "*Nne is an AI chatbot for pregnancy, postpartum, and maternal health. "
        "Ask questions below — for emergencies, always contact a healthcare provider.*"
    )

    with gr.Row():
        # ----- Left sidebar -----
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown(ABOUT_MARKDOWN)
            gr.Markdown(HOW_TO_USE_MARKDOWN)

        # ----- Main chat area -----
        with gr.Column(scale=3):
            gr.Markdown(f"**{TAGLINE}**")
            chatbot = gr.Chatbot(
                label="Chat with Nne",
                height=400,
                avatar_images=(
                    None,
                    "https://em-content.zobj.net/source/google/387/rose_1f339.png",
                ),
            )
            msg = gr.Textbox(
                label="Your question",
                placeholder=PLACEHOLDER,
                lines=2,
                max_lines=4,
                show_label=False,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask Nne", variant="primary")
                clear_btn = gr.Button("Clear Chat")
                export_btn = gr.Button("Export chat")
            export_file = gr.File(
                label="Download export", visible=True, interactive=False
            )

    chat_history = gr.State([])

    submit_btn.click(
        chat_submit, inputs=[msg, chat_history], outputs=[chatbot, msg, chat_history]
    ).then(lambda: None, None, [msg], queue=False)
    msg.submit(
        chat_submit, inputs=[msg, chat_history], outputs=[chatbot, msg, chat_history]
    ).then(lambda: None, None, [msg], queue=False)
    clear_btn.click(chat_clear, outputs=[chatbot, msg]).then(
        lambda: [], None, [chat_history], queue=False
    )
    export_btn.click(export_chat, inputs=[chat_history], outputs=[export_file])

    demo.load(fn=init_chat, outputs=[chatbot, chat_history])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=SHARE,
        theme=PINK_THEME,
        css=CUSTOM_CSS,
    )
