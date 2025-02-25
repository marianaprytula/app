import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "Marivanna27/fine-tuned-model_llama3_1_binary"


@st.cache_resource
def load_model():

        st.write("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HUGGINGFACE_TOKEN)

        st.write("⏳ Loading base model with 4-bit quantization (this may take time)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=HUGGINGFACE_TOKEN,
            device_map="auto"  # Auto-detect GPU if available
        )

        st.write("Loading LoRA adapter")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

        return model, tokenizer
    except Exception as e:
        st.error(f"🚨 Error loading model: {str(e)}")
        return None, None

# Load model & tokenizer
model, tokenizer = load_model()



st.title("LLaMA3 бінарна класифікація тональності тексту")
st.write("Введіть промпт та текст, який потрібно класифікувати. Відповідь 0 вказує, що текст негативний, а 1 - якщо текст позитивний.")
st.write()
st.write("🟡 Варіанти підказок для проведення бінарної класифікації тональності:")
st.write("""1️⃣ Визнач загальну тональність наступного коментаря, використовуючи лише одну з можливих цифр: 0, або 1. Відповідай 0, якщо коментар є негативним, відповідай 1, якщо коментар є позитивним. 
            Коментар:""")
st.write("""2️⃣ Ти спеціаліст з аналізу тональності текстів. 
        Якщо в тексті виявлено одночасно і позитивну, і негативну тональності, тоді визнач, яка з них є домінуючою і подай її як основний результат.
        Відповідай 0, якщо коментар є негативним, відповідай 1, якщо коментар є позитивним. 
        Формат відповіді: тільки одне число (0 або 1).
        Питання - Визнач загальну тональність коментаря: """)

user_input = st.text_area("✏️ Введіть промпт:", "")



if st.button("⏳Провести класифікацію"):
    if user_input:
        # Tokenize input
        input_ids = tokenizer(user_input, return_tensors="pt")


        outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 2)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Display the result
        st.write("### ✅ Тональність:")
        st.write(generated_text)
    else:
        st.warning("🚨Введіть промпт!")
