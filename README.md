# 🎨 Audio2Art – Voice-to-Image AI Generator

**Audio2Art** is an AI-powered application that transforms human voice prompts into stunning visual artworks.  
It combines cutting-edge speech recognition and image generation technologies to deliver an intuitive and creative experience.

---

## 🎥 Demo Video

📺 [Watch on Google Drive](https://drive.google.com/your-video-link) *(Optional)*

---

## 🛠️ Features

- 🎤 Converts **voice inputs** into text using **Wav2Vec2**
- 🖼️ Generates **high-quality artworks** from text using **Stable Diffusion**
- 🌐 Simple and interactive UI built with **Streamlit**
- 📦 Works locally or via **Google Colab + Localtunnel**
- ⚡ Modular design with separate logic and UI layers

---

## 🧠 Technologies Used

- **Python**
- **Hugging Face Transformers**
- **Wav2Vec2** – Speech Recognition
- **Stable Diffusion** – Text-to-Image Generation
- **Streamlit** – Web UI
- **Google Colab** / Local Deployment

---

## 📂 Project Structure

Audio2Art/
├── app.py # Streamlit UI

├── ImageModel.py # Core logic: promptgen + text2image

├── requirements.txt # Project dependencies

├── assets/ # Screenshots & demo images

├── docs/ # Project report & documentation

└── demo.mp4 # (Optional) Demo video

---

## 📷 Screenshots

### 🎤 Voice Input Prompt
<img src="assets/screenshot1.png" width="500"/>

### 🖼️ Generated Artwork Output
<img src="assets/screenshot2.png" width="500"/>

---

## 📄 Documentation

📝 Project report available here:  
📁 [`docs/Audio2Art_Project_Report.pdf`](docs/Audio2Art_Project_Report.pdf)

---

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
  git clone https://github.com/your-username/Audio2Art.git
  cd Audio2Art
```
2.Install dependencies
```
  pip install -r requirements.txt
```
3.Run the app
```
  streamlit run app.py
```
4.Access the app
  The terminal will show a local URL. Open it in your browser.

📜 License

This project is open-source under the MIT License.

