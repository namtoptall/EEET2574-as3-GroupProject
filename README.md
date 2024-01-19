# Group 4 EEET2574: Big Data for Engineering Assignment 3 Repository
## Problem Statement : 
The objective of this project is to identify regions with similar economic, deforestation, and weather patterns in order to provide tailored policy recommendations.

# Technology/Infrastructure/Models

## Technology Stack

### Programming Languages
- **Python:** Utilized for data analysis, manipulation, and visualization.
- **MongoDB:** Employed for efficient and scalable database operations.

### Libraries and Frameworks
- **Pandas:** Used for comprehensive data manipulation and analysis.
- **NumPy:** Essential for high-performance numerical computations.
- **Matplotlib and Seaborn:** Employed for creating insightful data visualizations.
- **Pymongo:** Facilitates MongoDB operations.
- **Scikit-learn:** Utilized for machine learning tasks, including model training and evaluation.
- **Boto3:** AWS SDK for Python, enabling interaction with AWS services.
- **SageMaker:** Leveraged for deploying and managing machine learning models on AWS.
- **os and system:** Python libraries for system operations.
- **Fitz:** Utilized for handling PDF documents in the data processing pipeline.

## Infrastructure

### Cloud Platform
- **Amazon Web Services (AWS):** Chosen for its robust cloud computing and storage capabilities.
  - **EC2:** Used for scalable compute capacity.
  - **SageMaker:** Employed for machine learning model hosting and management.

### Database
- **MongoDB:** Selected for its flexibility as a non-relational database, suitable for handling diverse data types.

### Streaming Data Processing
- **Apache Kafka (Confluent version):** Implemented for handling real-time streaming data, ensuring efficient and reliable processing.

## Models

### Decision Tree Regression
- **Purpose:** Analyzing and predicting economic trends.
- **Description:** Decision Tree Regression model employed for its interpretability and effectiveness in capturing non-linear relationships within the data.



## Data Sources

### Streaming Data
- Weather data sourced from the OpenWeather API.

### Saved Data and Files

- Vietnam deforestation data from the [Global Forest Watch website](https://www.globalforestwatch.org/map/?mainMap=eyJoaWRlTGVnZW5kIjp0cnVlLCJzaG93QW5hbHlzaXMiOnRydWV9&map=eyJjZW50ZXIiOnsibGF0IjoxNi4wMjgwMjY3MjEwNjM1NCwibG5nIjoxMDUuODA2OTAwMDAwMDAwOTV9LCJ6b29tIjo0LjYzMTg5MDk3NTA2NDc1OSwiY2FuQm91bmQiOmZhbHNlfQ%3D%3D&mapMenu=eyJzZWFyY2giOiJ2aWV0biJ9)

- Vietnam social-economic data from the [Vietnam General Statistics Office](https://wtocenter.vn/an-pham/22213-socio-economic-data-of-63-provinces-and-centrally-run-cities-2015-2021) (available as a PDF report).




## Data Storage
Our raw data is organized in the "/data/raw" directory:
- Raw deforestation dataset: "/data/raw/LoveWaterdata.xlsx"
- Raw Vietnam economic dataset: "/data/raw/socio-economic-data-of-63-provinces-and-centrally-run-cities-2015-2021.pdf"


## MongoDB Chart Visualization
Link: https://charts.mongodb.com/charts-bigdata-iewuu/public/dashboards/43633942-1da5-4522-8d1a-d5942341d20f
