# Evaluation Approaches for Retrieval Augmented Generation (RAG)

## About The Project

This is my Bachelor's thesis project at the University of Vienna, where I explore **evaluation techniques for Retrieval-Augmented Generation (RAG) systems**. The project leverages **Ollama** for running large language models locally, **R2R** for abstracting RAG workflows, and **Streamlit** for an interactive UI.

The primary goal is to assess different RAG evaluation frameworks, including **RAGAs**, to analyze how well retrieval enhances responses. Various of different **evaluation metrics** are to be used to achieve that.

### Built With

[![Python][Python-img]][Python-url] [![Docker][Docker-img]][Docker-url] [![Ollama][Ollama-img]][Ollama-url] [![R2R][R2R-img]][R2R-url] [![Streamlit][Streamlit-img]][Streamlit-url] [![RAGAs][RAGAs-img]][RAGAs-url] [![Unstructured][Unstructured-img]][Unstructured-url]

### Prerequisites

* Check if `python` is **installed** on your system. Make sure it is `3.12` and above.

```sh
python3 --version
```

* Ollama also needs to be locally available. If not run:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

* Finally, you will need to have docker. If not go to [Docker](https://www.docker.com/) and install the version appropriate for your OS.

### Structure

   ```sh
   .
   ├── docker-compose.yaml
   ├── env # Environment variables
   ├── project
   │   ├── backend # The main R2R Functionality
   │   │   ├── __init__.py
   │   │   ├── config.toml
   │   │   ├── conversations.py
   │   │   ├── documents.py
   │   │   ├── indices.py
   │   │   ├── prompts.py
   │   │   ├── retrieval.py
   │   │   └── settings.py
   │   ├── exports # One can export documents, conversations, etc
   │   ├── ragas # Evaluation using RAGAs
   │   │   ├── evaluate.ipynb
   │   │   ├── generate_dataset.ipynb
   │   ├── .streamlit
   │   ├── st_app.py
   │   ├── st_chat.py
   │   ├── st_conversation.py
   │   ├── st_index.py
   │   ├── st_prompt.py
   │   ├── st_settings.py
   │   └── st_storage.py
   ├── Readme.md
   ├── requirements.txt # All dependencies
   ├── run_r2r.sh
   ├── run_streamlit.sh
   └── services
      └── unstructured # Service for ingestion
         ├── Dockerfile
         └── main.py
   ```

### Usage

1. Clone the repo

   ```sh
   git clone https://github.com/danielpetrov18/Evaluation-Approaches-for-Retrieval-Augmented-Generation-RAG-
   ```

2. Switch into the directory

   ```sh
   cd Evaluation-Approaches-for-Retrieval-Augmented-Generation-RAG-
   ```

3. To start R2R

   ```sh
   chmod u+x run_r2r.sh
   # This will make sure that:
   # 1. Ollama has the required models and is running
   # 2. Docker is running
   # 3. Create a virtual environment and install dependencies
   # 4. Activate virtual environment
   # 5. Export environment variables
   # 6. Finally, start the r2r service
   ./run_r2r.sh   
   ```

4. To be able to interact with the backend over a GUI

   ```sh
   chmod u+x run_streamlit.sh
   # Assuming you have followed the previous step:
   # 1. Activate virtual environment
   # 2. Export environment variables
   # 3. Start the streamlit server
   # 4. Navigate to http://localhost:8501 in your browser
   ./run_streamlit.sh
   ```

5. For performing an evaluation. Example with RAGAs

   ```sh
   cd project && cd ragas
   # generate a dataset by using the generate_dataset notebook
   # thereafter select metrics https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
   # use the evaluate notebook with your selected metrics
   ```

### Features

* ✅ Local LLM Execution – Uses **Ollama** to run models locally
* ✅ RAG Pipeline Automation – **R2R** simplifies the RAG workflow
* ✅ **Streamlit** – Easy-to-use interface for interacting with the system
* ✅ Evaluation with **RAGAs** – Assessing RAG performance using multiple evaluation metrics
* ✅ Pre-built `Shell Scripts` – Automates model setup, environment configuration, and server startup

## Contact

Daniel Petrov - daniel.petrov18@protonmail.com

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python-url]: https://www.python.org/
[Python-img]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=black
[Docker-url]: https://www.docker.com/
[Docker-img]: https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white
[Ollama-url]: https://ollama.com/
[Ollama-img]: https://img.shields.io/badge/Ollama-ffffff.svg?style=for-the-badge&logo=ollama&logoColor=black&logoSize=auto
[R2R-url]: https://r2r-docs.sciphi.ai
[R2R-img]: https://img.shields.io/badge/R2R-black.svg?style=for-the-badge&logo=data:image/png;base64,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&logoColor=black
[Streamlit-url]: https://streamlit.io/
[Streamlit-img]: https://img.shields.io/badge/Streamlit-black.svg?style=for-the-badge&logo=streamlit&logoSize=auto&logoColor=red
[Unstructured-url]: https://unstructured.io/
[Unstructured-img]: https://img.shields.io/badge/Unstructured-white.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAFHUlEQVR4nOxYeWxU1Re+y5vpzHToQn8F8oPSlsUiXShFgbaAXZClggLBSgADRhGJNqGGKFRNUKxAiGENSAyLpQqExRZpWAoqpUhLpzVMEyTs1ikdFguIzLzt3mueo8Ob1zcMY0jaP+b7Z96c7yzfnHfeve8OZ7nMg64E1NkCtAgJCoSQoEAICQoEzh8hH6nEI8fAyGhy7iygFKcM9VLkfDNw/YkzMhW3mmrUPwn17quOJXYbbTgFEwbgvAIIocfIKCVHKrw+sE88Th3Wsa7fDonFc2jLFSX7vh3yri0+9Sq+EWaOJ011AADp00W0odaHra8RCnOp7ZS4+C1ycM9DghJpw3Jp+WJx4Rxpw3Ly/aHgOhQAsiQUzTJXNXRk6MVfgDWSe6fEmDgQWsK9dsgZzFUN8uFvpdL3dQM9+I8zhPIKYHSM8MHbgDENxb0wHaVmCFOy+MkjPF0MLrNfxmJlbQ7lF19vASazhoQmS9gXe2jdCXb1UodIZigqMdtaYXx/aePKJyYIT5giLnrd/eJIWv0dN+4lncg+8cZl6wGjGrt8rEqYkS+8+xq129DgIcEKgv52eyZJ5Gglu9GGnsnCaT6PA2luAm4XHj5KKX+0Eg1KQ30TfRzOnKRN9TAuAU+cChFWU7S1hTY3chOmBi2os9DlFsaQoEAICQoEv1uHVL4ZSCKwhOO8AhTbS02Rc2fZr5e5idOYLMtlm7hps2FUtI/DmZOkrgaGd8OTpqOe//fa5b1l7P49zzWM7cVNerljXb+PvSu9J4xLAPfuMELM1Xb1riTt2EQOV5i+PsJ43p0cZaq2o35PPWS3rZdK30NZuazNwdpvmw7Ued8F+OI57NplZrfBZ7NRUkrYx2uD6BAAwLh0DUpOd6fG0PPNOGPkIzy9YIRI60oNn6wzzJzHRFEu2wjQw6kwrf6KXrnAP59m2l4FTSbdDI8SJK0vBTwPukWifkmPo0bB3Xbwx12Unac032g0vLHwcQP/xaOGGnaPpRfPcfMXaUYEmMOZs5URwq63KF9VdxNEx4Du/yMHdivdenBfWFpMr1x4YoK4WW8aikrk8s1MFNR2PGYcu9vO56fwhXkoK1c9thAhQ8lKecNn7nHp7pynae0xEBkdlCC/Q01On0DJQ4AlnNbVoLRhMCJKzdLfb9H6GmC24Ox8aDRqYun13+jP9dAagTKfg8YwNcVcD2hTHcrMgRgHJ6iz0OUWxpCgQAgJCgT9lZredJLKnYZ5xcp6va8cp6SjpBQvS+prqL0RQIAGpeFR+drYW05SsdNzjbLzsOo9X/pytfIRFoZzJ6K4RN3S+h1iToe0YonnWt66ltptapb8cEjaspbUHhfmT5f2lnWIbZVWLFE2/MbT7KZTTSn2owfkfeX8+KHk9I9BCAoIPHy0aftBPHU2rT2u68C9usAwr5jLGa+xGz5aZa78CU9+RVyz7EkKImdO8nMnk/070IjRug7igkKhaJa/cJScDpytupT+DMFukco0tFyFsT3Z7RvAGql1iI6B4VYQ1Z0rnKubwdzggBYrANqDtnJP798jh/fDQan6WvUFJQxAORP4gmHuzERojcCjx2rDBg42fr4VQChX7tLN4M5KdKX3kHZv09iFGWPdQ3uxO+3GD1fpl/Z7cqWENtUDkUcZmdD3bE/bHEAUUXw/6rgGZIIS+vsEuh7QS+f/yd47DsX08FLE3vh3+yNgwgDv/0aPK6iz0OUWxpCgQAgJCoQuJ+ivAAAA//9XXP2G9+bddQAAAABJRU5ErkJggg==&logoSize=auto
[RAGAs-url]: https://docs.ragas.io/en/stable/
[RAGAs-img]: https://img.shields.io/badge/Ragas-black.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAADNElEQVR42u2avW4TQRDHc3EQ6QJEQXQ0vAMtFVJyuyDRUfHV8gRIiDQgKkQRg2h4BSQaoKQAFO8awkeAAl4CUST2eZa59f/sTc54t0jubiOvNJrx+s6Zn/93M7s5z00aRsnEei3m547SMEoksSU8b71OL3L8mZW5Ey0MLqm3RkvDyefWHs13IoIxm2vHOPlvFkSLXQA9iw7GvJcMIhnEqpGx7xsVFwxARK7INpQYkJJEMcJAkREIPOXKUDwwZRA28sFEo0gkMH6QSGD8IJHA+EGigvGDhMOQEklDQfwwpgSTJhGAlGDcpvnUXWhGAeLCwPex0HziwjQGJEoYP4gfhjwwtYFUAFNvH3GXMKQkhcGkSbM6uyrmRQZPbGEwWiSNUYSU2OX4D2KbLMfEMU0pAG0Xph6QclL3TSdd4oS28H4PMLlak2F0GaZOkAwg1+zx3XSZ40+A6E1ThpRE0wQM7pm6QW6Ov9nVJX790SapRY/9EEYDQvv7TOUgNAa5gSSOW9+RJzneyiGIYYbHACawadajCEBIiVZugDnF819G94wWuQ+rZoCpDcQAYuS1XObEv0IRC4ObPximUhDaB+LG1F1bYYBtoxxlUABCm2blN7sLgpLaQjU7zfEPVKmehdfTlaGSMocEQnsVue6CTIRR6RmOf9pztchQ1aiCVXNYQwTIbfyxRU5ugc9ZyL0TL6IYnOXXv1DNBuxDFpob7rb5EEByw9rKNsHVE4GPKy6gUQ6gaghM24U5UBDSgpBQNvTiN9tDUvIu+3Wev2dNCfZsSq6zz997ZZRTiv2bswxf2oZbAA5SEWJfwNjLBNe9Iesdw3xxDM6rbqeJ/8ZPBHHkJygyyLs4+gWbHBvmyc6LzKNIKEy4MrTn+YgESLXm388EwJgPVpHvjiJUKwyUoX0wpEUAjBLv8CEZW9Ug/moW/lRXPi92ge5N3hgYLR5PB9EAUeI8W/FBO2x9qFO39TmnHeR1C5dY6z8wlwqYq6zEX5TORpnb/Tk/C+KDOccqPWL/gu0N2+s6DTm8JCUemE2xgj3QtJsekjV7+CHcH9ewtUz3ctKY5LtXEuqIlu3yszEbsxE8/gFo+tDaHmgtXAAAAABJRU5ErkJggg==&logoSize=auto
