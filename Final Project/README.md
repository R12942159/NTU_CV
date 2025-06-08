# Iris Recognition Pipeline

This project implements an iris recognition pipeline utilizing a pre-trained deep learning model, evaluated on multiple benchmark datasets.

---
## Contributors
- 邱亮茗 (R12942159), Graduate Institute of Communication Engineering.
- 莊英博 (R13922A24), Department of Computer Science & Information Engineering.
- 王宥姍 (B11611001), Department of Biomechatronics Engineering.
- Anonymous

---
## Requirements  
- Python version: 3.13.3  
- Install dependencies: pip install -r requirements.txt

---
## Usage
- **Download datasets, use the following command:** <br>
    bash download_dataset.sh

- **Download pre-trained models, use the following command:** <br>
    bash download_models.sh

- **Run the iris recognition pipeline, use the following command:** <br>
    bash main.sh

- **Evaluate the recognition results (compute d' score), use the following command:** <br>
    bash evaluate.sh

---
## Project Structure
1. src: 
    Contains all source code related to this project. The main script is run.py.
    The directory src/modules/ includes the implementations related to iris recognition, with the core functionality encapsulated in the irisRecognition class (which can be referenced alongside run.py).

2. models:
    Stores the pre-trained model weights required for inference.

3. filters_pt:
    Contains pre-trained filter parameters.

4. input_list:
    Contains the input dataset file lists.

5. test:
    Stores the iris recognition results. Files named result_*.txt are the outputs submitted to Codabench.

6. cfg.yaml:
    Configuration file recording all necessary parameters for running the pipeline.

---
## References
For further details or clarifications on any part of this project, please refer to the following GitHub repositories:
- OpenSourceIrisRecognition main repository: https://github.com/CVRL/OpenSourceIrisRecognition
- Codebase referenced (minimal structural changes applied):
    - https://github.com/CVRL/OpenSourceIrisRecognition/tree/main/methods/TripletNN/Python
    - https://github.com/CVRL/OpenSourceIrisRecognition/tree/main/methods/HDBIF/Python