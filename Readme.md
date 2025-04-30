# Evaluation Approaches for Retrieval Augmented Generation (RAG)

## About The Project

This is my Bachelor's thesis project at the University of Vienna, where I explore **evaluation techniques for Retrieval-Augmented Generation (RAG) systems**. The project leverages **Ollama** for running large language models locally, **R2R** for abstracting RAG workflows, and **Streamlit** for an interactive UI. Additionally, **3 different frameworks for evaluation** are used.

The primary goal is to assess different RAG evaluation frameworks, including **RAGAs**, to analyze how well retrieval enhances responses. Various of different **evaluation metrics** are to be used to achieve that.

### Built With

[![Python][Python-img]][Python-url] [![Docker][Docker-img]][Docker-url] [![Ollama][Ollama-img]][Ollama-url] [![R2R][R2R-img]][R2R-url] [![Streamlit][Streamlit-img]][Streamlit-url] [![RAGAs][RAGAs-img]][RAGAs-url] [![DeepEval][DeepEval-img]][DeepEval-url]

### Prerequisites

* Check if `python` is **installed** on your system. Make sure it is at least `3.12`.

```sh
python3 --version
```

* Ollama also needs to be locally available. If not run:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

* Finally, you will need to have docker. If not go to [Docker](https://www.docker.com/) and install the version appropriate for your OS.

### Usage

* Make sure that `ufw - uncomplicated firewall` is turned of, if available on the system.

* Also make sure **Ollama** is available under `0.0.0.0`, otherwise even with `host.docker.internal` one cannot reach it from inside any of the containers. To modify it: `sudo nano /etc/systemd/system/ollama.service.`. Once modified use: `sudo systemctl reload ollama`. If that doesn't work try: `sudo systemctl reload-daemon`.

Example:

```bash
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=<path>
Environment="OLLAMA_DEBUG=1" # Optional
# Make sure it's available not only on localhost, otherwise R2R in container cannot reach it
Environment="OLLAMA_HOST=0.0.0.0"

[Install]
WantedBy=default.target
```

1. Clone the repo

   ```sh
   git clone https://github.com/danielpetrov18/Evaluation-Approaches-for-Retrieval-Augmented-Generation-RAG-
   ```

2. Switch into the directory

   ```sh
   cd Evaluation-Approaches-for-Retrieval-Augmented-Generation-RAG-
   ```

3. To start the application

   ```sh
   chmod u+x run.sh
   # This will make sure that:
   #  1. All environment variables are exported.
   #  2. Ollama is running and has the proper models installed.
   #  3. Docker is running and start all containers.
   #  4. Nvidia container toolkit is available, if not download it
   ./run.sh   
   ```

4. For performing an evaluation. Example with **RAGAs**

   ```sh
   # From root of project
   cd evaluation/ragas
   # To generate a synthetic dataset:
   #  -> Either use RAGAs
   #  -> or what I do is generate it using DeepEval
   # Make sure to run the `setup.sh` script to install all dependencies
   # Select `eval` as your kernel in the notebook
   # thereafter select metrics https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/ or use all the metrics I've used
   # The evaluation for the other frameworks is very similar (using a notebook with code + explanation)
   ```

### Features

* ✅ Local LLM Execution – Uses **Ollama** to run models locally
* ✅ RAG Pipeline Automation – **R2R** simplifies the RAG workflow
* ✅ **Streamlit** – Easy-to-use interface for interacting with the system
* ✅ Evaluation with **RAGAs**, **DeepEval** – Assessing RAG performance using multiple evaluation metrics and synthetic data generation
* ✅ Pre-built `Shell Script` – Automates model setup, environment configuration, and server startup

## Contact

Daniel Petrov - <daniel.petrov18@protonmail.com>

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
[RAGAs-url]: https://docs.ragas.io/en/stable/
[RAGAs-img]: https://img.shields.io/badge/Ragas-black.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAADNElEQVR42u2avW4TQRDHc3EQ6QJEQXQ0vAMtFVJyuyDRUfHV8gRIiDQgKkQRg2h4BSQaoKQAFO8awkeAAl4CUST2eZa59f/sTc54t0jubiOvNJrx+s6Zn/93M7s5z00aRsnEei3m547SMEoksSU8b71OL3L8mZW5Ey0MLqm3RkvDyefWHs13IoIxm2vHOPlvFkSLXQA9iw7GvJcMIhnEqpGx7xsVFwxARK7INpQYkJJEMcJAkREIPOXKUDwwZRA28sFEo0gkMH6QSGD8IJHA+EGigvGDhMOQEklDQfwwpgSTJhGAlGDcpvnUXWhGAeLCwPex0HziwjQGJEoYP4gfhjwwtYFUAFNvH3GXMKQkhcGkSbM6uyrmRQZPbGEwWiSNUYSU2OX4D2KbLMfEMU0pAG0Xph6QclL3TSdd4oS28H4PMLlak2F0GaZOkAwg1+zx3XSZ40+A6E1ThpRE0wQM7pm6QW6Ov9nVJX790SapRY/9EEYDQvv7TOUgNAa5gSSOW9+RJzneyiGIYYbHACawadajCEBIiVZugDnF819G94wWuQ+rZoCpDcQAYuS1XObEv0IRC4ObPximUhDaB+LG1F1bYYBtoxxlUABCm2blN7sLgpLaQjU7zfEPVKmehdfTlaGSMocEQnsVue6CTIRR6RmOf9pztchQ1aiCVXNYQwTIbfyxRU5ugc9ZyL0TL6IYnOXXv1DNBuxDFpob7rb5EEByw9rKNsHVE4GPKy6gUQ6gaghM24U5UBDSgpBQNvTiN9tDUvIu+3Wev2dNCfZsSq6zz997ZZRTiv2bswxf2oZbAA5SEWJfwNjLBNe9Iesdw3xxDM6rbqeJ/8ZPBHHkJygyyLs4+gWbHBvmyc6LzKNIKEy4MrTn+YgESLXm388EwJgPVpHvjiJUKwyUoX0wpEUAjBLv8CEZW9Ug/moW/lRXPi92ge5N3hgYLR5PB9EAUeI8W/FBO2x9qFO39TmnHeR1C5dY6z8wlwqYq6zEX5TORpnb/Tk/C+KDOccqPWL/gu0N2+s6DTm8JCUemE2xgj3QtJsekjV7+CHcH9ewtUz3ctKY5LtXEuqIlu3yszEbsxE8/gFo+tDaHmgtXAAAAABJRU5ErkJggg==&logoSize=auto
[DeepEval-url]: https://docs.confident-ai.com/docs/getting-started
[DeepEval-img]: https://img.shields.io/badge/DeepEval-black.svg?style=for-the-badge&logo=data:image/png;base64,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&logoSize=auto