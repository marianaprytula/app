import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

MODEL_NAME = "Marivanna27/fine-tuned-model_llama3_1_binary"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

# Load model & tokenizer
model, tokenizer = load_model()

# Streamlit UI
st.title("LLaMA3 бінарна класифікація тональності тексту")
st.write("Введіть промпт та текст, який потрібно класифікувати. Відповідь 0 вказує, що текст негативний, а 1 - якщо текст позитивний.")
st.write("Варіанти пропмта для проведення бінарної класифікації тональностей:")
st.write("""Визнач загальну тональність наступного коментаря, використовуючи лише одну з можливих цифр: 0, або 1. Відповідай 0, якщо коментар є негативним, відповідай 1, якщо коментар є позитивним. 
            Коментар:""")
st.write("""Ти спеціаліст з аналізу тональності текстів. 
        Якщо в тексті виявлено одночасно і позитивну, і негативну тональності, тоді визнач, яка з них є домінуючою і подай її як основний результат.
        Відповідай 0, якщо коментар є негативним, відповідай 1, якщо коментар є позитивним. 
        Формат відповіді: тільки одне число (0 або 1).
        Питання - Визнач загальну тональність коментаря: """)



user_input = st.text_area("Введіть промпт:", "")

if st.button("Провести класифікацію"):
    if user_input:
        # Tokenize input
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device)

        # Generate output
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=100, top_p=0.9, temperature=0.7)

        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Display the result
        st.write("### Тональність:")
        st.write(generated_text)
    else:
        st.warning("Введіть промпт!")
