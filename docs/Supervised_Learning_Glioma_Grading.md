# Supervised Learning - Glioma Grading Clinical and Mutation Features

![image](../images/brain-glioma.jpg)

---

## Group Members (Class 9 Group 2)

| Name | Student Number | Email |
| --- | --- | --- |
| [João Ramos](https://github.com/jfoliveiraramos) | 202108743 | up202108743@up.pt |
| [Marco Costa](https://github.com/SpardaMarco) | 202108821 | up202108821@up.pt |
| [Tiago Viana](https://github.com/tiagofcviana) | 201807126 | up201807126@up.pt |

---

## Introduction

**Gliomas** are the most common primary brain tumors.
Based on histological/imaging criteria, they can be classified as:
- LGG (Lower-Grade Glioma)
- GBM (Glioblastoma Multiforme)

For the grading process, clinical and molecular/mutation factors are highly important, and molecular tests for accurately diagnosing glioma patients are costly.

This is a **supervised learning** problem where the main goal is to leverage classification algorithms to grade gliomas based on clinical and genetic mutation features.
More specifically, we are trying to determine whether a glioma patient has **LGG** (Lower-Grade Glioma) or **GBM** (Glioblastoma Multiforme). 

Additionally, we are also trying to find the optimal subset of mutation genes and clinical features for the glioma grading process to improve performance and reduce costs.

The given dataset, from [Kaggle's Glioma Grading Clinical and Mutation Features](https://www.kaggle.com/datasets/vinayjose/glioma-grading-clinical-and-mutation-features), contains 862 records of patients who have brain glioma. Each record is characterized by **20** molecular features, each of which can be *mutated* or *not_mutated*, and **3** clinical features. Addionally, the dataset contains **2** features used to identify the original patient and **1** feature regarding the primary diagonsis given to the entry.

---

### Packages/Libraries:

- NumPy (1.26.0)
- MatPlotLib (3.8.4)
- Seaborn (0.13.2)
- Pandas (2.2.2)
- SciKit-Learn (1.4.2)

We used python 3.10.12 to develop this project.
To install the required packages/libraries, run:


```python
%pip install -r "../requirements.txt"
```

    Requirement already satisfied: numpy==1.26.0 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from -r ../requirements.txt (line 1)) (1.26.0)
    Requirement already satisfied: matplotlib==3.8.4 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from -r ../requirements.txt (line 2)) (3.8.4)
    Requirement already satisfied: seaborn==0.13.2 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from -r ../requirements.txt (line 3)) (0.13.2)
    Requirement already satisfied: pandas==2.2.2 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from -r ../requirements.txt (line 4)) (2.2.2)
    Requirement already satisfied: scikit-learn==1.4.2 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from -r ../requirements.txt (line 5)) (1.4.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (1.4.5)
    Requirement already satisfied: pillow>=8 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (10.2.0)
    Requirement already satisfied: packaging>=20.0 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (24.0)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (1.2.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (3.1.2)
    Requirement already satisfied: cycler>=0.10 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (4.50.0)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from matplotlib==3.8.4->-r ../requirements.txt (line 2)) (2.8.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from pandas==2.2.2->-r ../requirements.txt (line 4)) (2024.1)
    Requirement already satisfied: pytz>=2020.1 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from pandas==2.2.2->-r ../requirements.txt (line 4)) (2022.6)
    Requirement already satisfied: scipy>=1.6.0 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from scikit-learn==1.4.2->-r ../requirements.txt (line 5)) (1.13.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from scikit-learn==1.4.2->-r ../requirements.txt (line 5)) (3.4.0)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from scikit-learn==1.4.2->-r ../requirements.txt (line 5)) (1.4.0)
    Requirement already satisfied: six>=1.5 in c:\users\tfili\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from python-dateutil>=2.7->matplotlib==3.8.4->-r ../requirements.txt (line 2)) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip available: 22.3.1 -> 24.0
    [notice] To update, run: C:\Users\tfili\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\python.exe -m pip install --upgrade pip
    

---

## Data pre-processing

###  Data analysis

#### Create the dataframe


```python
import pandas as pd

data = pd.read_csv("../data/TCGA_GBM_LGG_Mutations_all.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grade</th>
      <th>Project</th>
      <th>Case_ID</th>
      <th>Gender</th>
      <th>Age_at_diagnosis</th>
      <th>Primary_Diagnosis</th>
      <th>Race</th>
      <th>IDH1</th>
      <th>TP53</th>
      <th>ATRX</th>
      <th>...</th>
      <th>FUBP1</th>
      <th>RB1</th>
      <th>NOTCH1</th>
      <th>BCOR</th>
      <th>CSMD3</th>
      <th>SMARCA4</th>
      <th>GRIN2A</th>
      <th>IDH2</th>
      <th>FAT4</th>
      <th>PDGFRA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LGG</td>
      <td>TCGA-LGG</td>
      <td>TCGA-DU-8164</td>
      <td>Male</td>
      <td>51 years 108 days</td>
      <td>Oligodendroglioma, NOS</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LGG</td>
      <td>TCGA-LGG</td>
      <td>TCGA-QH-A6CY</td>
      <td>Male</td>
      <td>38 years 261 days</td>
      <td>Mixed glioma</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LGG</td>
      <td>TCGA-LGG</td>
      <td>TCGA-HW-A5KM</td>
      <td>Male</td>
      <td>35 years 62 days</td>
      <td>Astrocytoma, NOS</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LGG</td>
      <td>TCGA-LGG</td>
      <td>TCGA-E1-A7YE</td>
      <td>Female</td>
      <td>32 years 283 days</td>
      <td>Astrocytoma, anaplastic</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LGG</td>
      <td>TCGA-LGG</td>
      <td>TCGA-S9-A6WG</td>
      <td>Male</td>
      <td>31 years 187 days</td>
      <td>Astrocytoma, anaplastic</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



#### Descriptive statistics

Before we started the data pre-processing, we performed some data analysis to understand the dataset better.


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grade</th>
      <th>Project</th>
      <th>Case_ID</th>
      <th>Gender</th>
      <th>Age_at_diagnosis</th>
      <th>Primary_Diagnosis</th>
      <th>Race</th>
      <th>IDH1</th>
      <th>TP53</th>
      <th>ATRX</th>
      <th>...</th>
      <th>FUBP1</th>
      <th>RB1</th>
      <th>NOTCH1</th>
      <th>BCOR</th>
      <th>CSMD3</th>
      <th>SMARCA4</th>
      <th>GRIN2A</th>
      <th>IDH2</th>
      <th>FAT4</th>
      <th>PDGFRA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>...</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
      <td>862</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>862</td>
      <td>3</td>
      <td>838</td>
      <td>7</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>LGG</td>
      <td>TCGA-LGG</td>
      <td>TCGA-DU-8164</td>
      <td>Male</td>
      <td>--</td>
      <td>Glioblastoma</td>
      <td>white</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>499</td>
      <td>499</td>
      <td>1</td>
      <td>499</td>
      <td>5</td>
      <td>360</td>
      <td>766</td>
      <td>448</td>
      <td>508</td>
      <td>642</td>
      <td>...</td>
      <td>815</td>
      <td>821</td>
      <td>824</td>
      <td>833</td>
      <td>834</td>
      <td>834</td>
      <td>835</td>
      <td>839</td>
      <td>839</td>
      <td>840</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 27 columns</p>
</div>




```python
print(f"The dataset contains {data.shape[0]} entries and {data.shape[1]-1} features.")
```

    The dataset contains 862 entries and 26 features.
    

### Removal of redundant columns

The columns "Project", "Case_ID" and "Primary_Diagnosis" of the original dataset are not relevant for the classification task and will be removed.

The "Project" column identifies the name of the project, which is the same for all records, and the grade of the glioma, which is already present in the "Grade" column. The "Case_ID" is a unique identifier for each record, and the "Primary_Diagnosis" column corresponds to the primary diagnosis given to the patient, which won't be considered for the classification task.


```python
data = data.set_index(data["Case_ID"])
data = data.drop(columns=["Case_ID", "Project", "Primary_Diagnosis"])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grade</th>
      <th>Gender</th>
      <th>Age_at_diagnosis</th>
      <th>Race</th>
      <th>IDH1</th>
      <th>TP53</th>
      <th>ATRX</th>
      <th>PTEN</th>
      <th>EGFR</th>
      <th>CIC</th>
      <th>...</th>
      <th>FUBP1</th>
      <th>RB1</th>
      <th>NOTCH1</th>
      <th>BCOR</th>
      <th>CSMD3</th>
      <th>SMARCA4</th>
      <th>GRIN2A</th>
      <th>IDH2</th>
      <th>FAT4</th>
      <th>PDGFRA</th>
    </tr>
    <tr>
      <th>Case_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TCGA-DU-8164</th>
      <td>LGG</td>
      <td>Male</td>
      <td>51 years 108 days</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-QH-A6CY</th>
      <td>LGG</td>
      <td>Male</td>
      <td>38 years 261 days</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-HW-A5KM</th>
      <td>LGG</td>
      <td>Male</td>
      <td>35 years 62 days</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-E1-A7YE</th>
      <td>LGG</td>
      <td>Female</td>
      <td>32 years 283 days</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-S9-A6WG</th>
      <td>LGG</td>
      <td>Male</td>
      <td>31 years 187 days</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### Removal of invalid entries

Since the values in the "Grade" column and in all the columns that represent the genetic mutations are categorical and binary, we checked the unique values of each of those columns for invalid entries.


```python
columns = []

## Check if the target column respects the binary categorization
if len(data["Grade"].unique()) != 2:
    columns.append("Grade")

## Check if the genetic mutations columns respect the binary categorization
for column in data.columns[4:24]:
    if len(data[column].unique()) != 2:
        columns.append(column)

if len(columns) == 0:
    print('The "Grade" and genetic mutations columns all have 2 unique values.')

else:
    for column in columns:
        print(f'"{column}" does not respect the binary categorization.')
        print(f'Unique values in "{column}": {data[column].unique()}\n')
```

    The "Grade" and genetic mutations columns all have 2 unique values.
    

As the values in the "Gender" and the "Race" columns are also categorical, we also checked the unique values of those columns for invalid entries.


```python
print(f'Unique values in "Gender": {data["Gender"].unique()}')
print(f'Unique values in "Race": {data["Race"].unique()}')
```

    Unique values in "Gender": ['Male' 'Female' '--']
    Unique values in "Race": ['white' 'asian' 'black or african american' '--' 'not reported'
     'american indian or alaska native']
    

To facilitate the analysis, we will convert the entries in the "Age_at_diagnosis" column to float.


```python
def convert_age_to_float(duration):
    years = 0
    days = 0
    
    parts = duration.split()
    
    for i in range(len(parts)):
        if parts[i] == 'years':
            
            years = float(parts[i - 1])
        elif parts[i] == 'days':
            
            days = float(parts[i - 1])
        
    return years + days / 365.25

def normalize_age(age):
    return round(age/100, 3)

data['Age_at_diagnosis'] = data['Age_at_diagnosis'].apply(convert_age_to_float)
data['Age_at_diagnosis'] = data['Age_at_diagnosis'].apply(normalize_age)

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grade</th>
      <th>Gender</th>
      <th>Age_at_diagnosis</th>
      <th>Race</th>
      <th>IDH1</th>
      <th>TP53</th>
      <th>ATRX</th>
      <th>PTEN</th>
      <th>EGFR</th>
      <th>CIC</th>
      <th>...</th>
      <th>FUBP1</th>
      <th>RB1</th>
      <th>NOTCH1</th>
      <th>BCOR</th>
      <th>CSMD3</th>
      <th>SMARCA4</th>
      <th>GRIN2A</th>
      <th>IDH2</th>
      <th>FAT4</th>
      <th>PDGFRA</th>
    </tr>
    <tr>
      <th>Case_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TCGA-DU-8164</th>
      <td>LGG</td>
      <td>Male</td>
      <td>0.513</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-QH-A6CY</th>
      <td>LGG</td>
      <td>Male</td>
      <td>0.387</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-HW-A5KM</th>
      <td>LGG</td>
      <td>Male</td>
      <td>0.352</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-E1-A7YE</th>
      <td>LGG</td>
      <td>Female</td>
      <td>0.328</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
    <tr>
      <th>TCGA-S9-A6WG</th>
      <td>LGG</td>
      <td>Male</td>
      <td>0.315</td>
      <td>white</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>...</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
      <td>NOT_MUTATED</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
invalid_entries = 0

for age in data['Age_at_diagnosis']:
    if age <= 0:
        invalid_entries += 1

if invalid_entries == 0:
    print('All entries in "Age_at_diagnosis" are valid.')

else:
    print(f'There are {invalid_entries} invalid entries in "Age_at_diagnosis".')
```

    There are 5 invalid entries in "Age_at_diagnosis".
    

After this primary analysis, we noted that some entries in the dataset were invalid, so we removed them.


```python
import numpy as np

print(f"{data.shape[0]} entries before removal.")
data = data.replace(['--'], np.nan)
data = data.replace(['not reported'], np.nan)
data["Age_at_diagnosis"] = data["Age_at_diagnosis"].replace(0, np.nan)
data = data.dropna()
print(f"{data.shape[0]} entries after removal.")

```

    862 entries before removal.
    839 entries after removal.
    

### Converting entries to numerical types

To perform the classification task, we needed to convert the categorical entries to numerical types. We will use the following mappings:

| Grade    |   |
|:---------|---|
| 'LGG'    | 0 |
| 'GBM'    | 1 |

<br>

| Gender   |   |
|:---------|---|
| 'Male'   | 0 |
| 'Female' | 1 |

<br>

| Race                               |    |
|:-----------------------------------|----|
| 'white'                            |  0 |
| 'black or african american'        |  1 |
| 'asian'                            |  2 |
| 'american indian or alaska native' |  3 |
 
<br>

| Genes         |   |
|:--------------|---|
| 'NOT_MUTATED' | 0 |
| 'MUTATED'     | 1 |


```python
pd.set_option('future.no_silent_downcasting', True)

data = data.replace({'LGG': 0, 'GBM': 1});
data = data.replace({'NOT_MUTATED': 0, 'MUTATED': 1});
data['Race'] = data['Race'].replace({'white': 0, 'black or african american': 1, 'asian': 2, 'american indian or alaska native': 3});
data = data.replace({'Male': 0, 'Female': 1});
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grade</th>
      <th>Gender</th>
      <th>Age_at_diagnosis</th>
      <th>Race</th>
      <th>IDH1</th>
      <th>TP53</th>
      <th>ATRX</th>
      <th>PTEN</th>
      <th>EGFR</th>
      <th>CIC</th>
      <th>...</th>
      <th>FUBP1</th>
      <th>RB1</th>
      <th>NOTCH1</th>
      <th>BCOR</th>
      <th>CSMD3</th>
      <th>SMARCA4</th>
      <th>GRIN2A</th>
      <th>IDH2</th>
      <th>FAT4</th>
      <th>PDGFRA</th>
    </tr>
    <tr>
      <th>Case_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TCGA-DU-8164</th>
      <td>0</td>
      <td>0</td>
      <td>0.513</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TCGA-QH-A6CY</th>
      <td>0</td>
      <td>0</td>
      <td>0.387</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TCGA-HW-A5KM</th>
      <td>0</td>
      <td>0</td>
      <td>0.352</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TCGA-E1-A7YE</th>
      <td>0</td>
      <td>1</td>
      <td>0.328</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TCGA-S9-A6WG</th>
      <td>0</td>
      <td>0</td>
      <td>0.315</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
print(f"The clean dataset contains {data.shape[0]} entries and {data.shape[1]-1} features.")
```

    The clean dataset contains 839 entries and 23 features.
    

### Feature Extraction

Here, we aim to perform feature extraction to reduce the dimensionality of the dataset and improve the performance of the classification algorithms.

We will deduce the redundancy of the features by using the Pearson correlation coefficient and discard the features that are highly correlated with each other.


```python
import math
import matplotlib.pyplot as plt

cols = list(data.columns)

plot_cols = []

for i, col1 in enumerate(cols):
    if col1 == 'Grade':
        continue
    for col2 in cols[i::]:
        if col1 == col2 or col2 == 'Grade':
            continue;
        if math.fabs(data[col1].corr(data[col2])) > 0.6:
            plot_cols.append([col1,col2])

if len(plot_cols) == 0:
    print('No features have a correlation greater than 0.6.')

else:
    plt.figure(figsize=(25,25))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
    for i, cols in enumerate(plot_cols):
        plt.subplot(4,4,i+1)
        plt.scatter(data[cols[0]],data[cols[1]],s=10,c='red',alpha=0.4)
        plt.xlabel(f"{cols[0]}",fontsize=12)
        plt.ylabel(f"{cols[1]}",fontsize=12)
    plt.show();
```

    No features have a correlation greater than 0.6.
    

As no pair of features presented a correlation coefficient higher than even 0.6, we concluded there is no high correlation between the features present in the dataset.

Likewise, we opted not to discard any features from the dataset, as none proved to be redundant.

### Correlation between each feature and the target variable

To understand the correlation between each feature and the target variable, we calculated the Pearson correlation coefficient.

We intended to determine which features appear to be the most relevant for the classification task.


```python
import seaborn as sb

sb.heatmap(data.corr()[['Grade']].drop('Grade').T, annot=False, cmap='seismic', center=0);
plt.title('Correlation between features and target');
```


    
![png](../images/output_33_0.png)
    


We were able to see that mutations of the genes "IDH1", "ATRX" and "CIC" are good indicators of **LGG**, with "IDH1" presenting a particularly higher correlation than the others.

On the other hand, we could notice that the higher the "Age at diagnosis", the higher the probability of the patient's glioma being graded as **GBM**. 

### Checking for outliers

Since the "Age at diagnosis" feature is the only numerical feature in the dataset, we will check for outliers in this feature. The features corresponding to the genetic mutation all have binary values, so there is no need to check for outliers.


```python
sb.histplot(data['Age_at_diagnosis'], kde=True);
plt.title('Age at diagnosis distribution');
plt.xlabel('Age at diagnosis');
```


    
![png](../images/output_37_0.png)
    


By analysing the histogram of this feature, we can verify that there are no outliers. Considering the normalization of the data performed before-hand, we can see that the values are within a reasonable range (i.e., between close to 10 years old and 90 years old).

### Data visualization

To understand if the data is balanced or not, we will plot the distribution of the classes in the "Grade" column, which represents the glioma grade.


```python
sb.displot(data['Grade'], stat='percent', discrete=True, shrink=0.8);
plt.xticks([0, 1], ['LGG', 'GBM']);
plt.title('Grade distribution');
```


    
![png](../images/output_41_0.png)
    


By the analysis of the plot, we could see that the data was, in fact, balanced, with a similar number of records for each class.


```python
sb.displot(data=data, x="Race", hue="Grade", binwidth=0.3);
plt.xticks(sorted(data["Race"].unique()), ["White", "Black", "Asian", "Native"]);
plt.title('Distribution of race by grade');
sb.displot(data=data, x="Gender", hue="Grade", binwidth=0.3);
plt.xticks(sorted(data["Gender"].unique()), ["Male", "Female"]);
plt.title('Distribution of gender by grade');
data_age_denormalized = data.copy()
data_age_denormalized["Age_at_diagnosis"] *= 100
sb.displot(data=data_age_denormalized, x="Age_at_diagnosis", hue="Grade");
plt.title('Distribution of age by grade');
plt.xlabel('Age at diagnosis');
```


    
![png](../images/output_43_0.png)
    



    
![png](../images/output_43_1.png)
    



    
![png](../images/output_43_2.png)
    


By analysing the distribution of the Glioma grade across "asian" and "american indian or alaska native" races, we noticed that these were under represented and lacked in the number and variety of data samples. Considering this, we decided to merge both into one single category, "other".


```python
data['Race'] = data['Race'].replace({3: 2});
```


```python
sb.displot(data=data, x="Race", hue="Grade", binwidth=0.3);
plt.xticks(sorted(data["Race"].unique()), ["White", "Black", "Other"]);
plt.title('Distribution of race by grade');
```


    
![png](../images/output_46_0.png)
    


Another viable approach would be to also merge the "black or african american" category with the "other" category, as they are also under represented, performing some operation to try and balance the data (e.g., duplicating the data in these categories). However, in order to maintain the diversity of the data, we decided to keep them separated. Given the nature of the dataset, even though both these categories were under represented, there still can be an interest in the differences between them and the "white" category.

### Exporting clean dataset file


```python
data.to_csv("../data/TCGA_GBM_LGG_Mutations_clean.csv", index=True)
```

---

## Supervised Learning Algorithms

In this section, we will apply different supervised learning algorithms to the dataset, obtain the best model for each algorithm, and evaluate the performance of each selected model.

For the sake of the reproducibility of the following process, we will use a fixed random state for the algorithms that have a random component.


```python
random_state = 1
```

### Data Splitting

We will split the dataset into training and test sets, with **80%** of the data being used for training and **20%** for testing.

The training set will be used to train and fine-tune the different models, obtaining the best set of hyperparameters for each one.

The test set will be used to evaluate and compare the different performance metrics across each selected model. 


```python
from sklearn.model_selection import train_test_split

all_labels = data['Grade'].astype(int).values
all_inputs = data.drop(columns=['Grade']).values

X_train, X_test, y_train, y_test = train_test_split(all_inputs, all_labels, test_size=0.2, random_state=random_state)
```

### Evaluating Classification Models

For this classification task, we will train each model according to the following classification algorithms, as provided by the scikit-learn library:
- Nearest Neighbors
- Decision Tree
- Support Vector Machine
- Neural Network
- Guassian Naive Bayes
- Random Forest
- Gradient Boosting


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold

nearest_neighbors_classifier = KNeighborsClassifier()
decision_tree_classifier = DecisionTreeClassifier()
support_vector_classifier = SVC()
neural_network_classifier = MLPClassifier()
naive_bayes_classifier = GaussianNB()
random_forest_classifier = RandomForestClassifier()
gradient_boosting_classifier = GradientBoostingClassifier()

selected_models = {}

cross_validation = StratifiedKFold(n_splits=5)

```

In order to save the trained models and reuse them later for predictions, we will use the `joblib` library.

As some of the algorithms take a long time to train the models (Gradient Boosting takes up to 1 hour), we will a add a flag `load_models` to enable the loading of the trained models from the disk.

If the flag is set to `False`, the models will be trained and saved to the disk. If the flag is set to `True`, the models will be loaded from the disk and the training process will be skipped.

For the purpose of demonstration, we pre-trained the models using grid search and saved them to the disk. The models are saved in the `models` folder.


```python
import joblib

load_models = True
```

#### Nearest Neighbors


```python
if not load_models:

    from sklearn.model_selection import GridSearchCV

    parameter_grid = {'n_neighbors': range(1, 10),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': range(10, 50, 10),
                    'p': range(1, 5)}

    nn_grid_search = GridSearchCV(estimator=nearest_neighbors_classifier,
                            param_grid=parameter_grid,
                            cv=cross_validation,
                            n_jobs=-1)

    nn_grid_search.fit(X_train, y_train)

    selected_models['Nearest Neighbors'] = nn_grid_search.best_estimator_

    joblib.dump(nn_grid_search.best_estimator_, '../models/nearest_neighbors.pkl')

    print('Best score: {}'.format(nn_grid_search.best_score_))
    print('Best parameters: {}'.format(nn_grid_search.best_params_))
    print('Training time: {}'.format(nn_grid_search.refit_time_))
```


```python
if load_models:
    selected_models['Nearest Neighbors'] = joblib.load('../models/nearest_neighbors.pkl')
```

#### Decision Tree


```python
if not load_models:

    parameter_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_depth': range(1, 23),
                    'min_samples_split': range(2, 10),
                    'min_samples_leaf': range(1, 5),
                    'ccp_alpha': [0.0, 0.1, 0.2],
                    'random_state': [random_state]}

    dt_grid_search = GridSearchCV(estimator=decision_tree_classifier,
                                param_grid=parameter_grid,
                                cv=cross_validation,
                                n_jobs=-1)

    dt_grid_search.fit(X_train, y_train)

    selected_models['Decision Tree'] = dt_grid_search.best_estimator_

    joblib.dump(dt_grid_search.best_estimator_, '../models/decision_tree.pkl')

    print('Best score: {}'.format(dt_grid_search.best_score_))
    print('Best parameters: {}'.format(dt_grid_search.best_params_))
    print('Training time: {}'.format(dt_grid_search.refit_time_))
```


```python
if load_models:
    selected_models['Decision Tree'] = joblib.load('../models/decision_tree.pkl')
```

##### Exporting the decision tree to a file


```python
import sklearn.tree as tree

with open('../out/glioma_decision_tree_classifier.dot', 'w') as out_file:
    out_file = tree.export_graphviz(selected_models['Decision Tree'], out_file=out_file, feature_names=data.columns[1:])
```

#### Support Vector Machine


```python
if not load_models:

    parameter_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': range(2, 5),
                    'gamma': ['scale', 'auto'],
                    'random_state': [random_state]}

    svm_grid_search = GridSearchCV(estimator=support_vector_classifier,
                                param_grid=parameter_grid,
                                cv=cross_validation,
                                n_jobs=-1)

    svm_grid_search.fit(X_train, y_train)

    selected_models['Support Vector'] = svm_grid_search.best_estimator_

    joblib.dump(svm_grid_search.best_estimator_, '../models/support_vector.pkl')

    print('Best score: {}'.format(svm_grid_search.best_score_))
    print('Best parameters: {}'.format(svm_grid_search.best_params_))
    print('Training time: {}'.format(svm_grid_search.refit_time_))
```


```python
if load_models:
    selected_models['Support Vector'] = joblib.load('../models/support_vector.pkl')
```

#### Neural Network


```python
def display_mlp_loss_curve(mlp_model):
    plt.figure(figsize=(10, 5))
    plt.title("Convergence of loss function")
    plt.plot(mlp_model.loss_curve_)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    
def display_mlp_weights(mlp):
    fig = plt.figure(figsize=(20, 10))

    for i, coef in enumerate(mlp.coefs_):
        ax = fig.add_subplot(2, mlp.n_layers_  // 2, i + 1)
        fig.tight_layout(w_pad=4.0, h_pad=5.0)
        im = ax.imshow(coef, aspect='auto', cmap='coolwarm')
        ax.set_title(f'Layer {i+1}')
        ax.set_yticks(range(coef.shape[0]))
        if i == 0:
            ax.set_yticklabels(data.columns[1:])
            ax.set_ylabel('Feature')
        else:
            ax.set_ylabel(f'Layer {i} Neuron')
            if coef.shape[0] > 25:
                ax.set_yticklabels([])
                
        ax.set_xticks(range(coef.shape[1]))
        if coef.shape[1] > 25:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(range(coef.shape[1]))
        ax.set_xlabel('Neuron')

    fig.colorbar(im, orientation='vertical', fraction=.1)
    plt.show()
```


```python
if not load_models:

  hidden_layer_sizes = [(50,), (25,), (10,), (50, 25), (25, 10), (50, 25, 10), (25, 10, 5), (50, 25, 10, 5), (25, 10, 5, 2), (50, 25, 10, 5, 2)]

  parameter_grid = {'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': ['identity', 'logistic', 'tanh'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'batch_size': range(20, 100, 20),
                    'learning_rate': ['constant', 'adaptive'],
                    'early_stopping': [True, False],
                      'max_iter': [1000], 
                    'random_state': [random_state]}
                                                          
  mlp_grid_search = GridSearchCV(estimator=neural_network_classifier,
                              param_grid=parameter_grid,
                              cv=cross_validation,
                              n_jobs=-1)

  mlp_grid_search.fit(X_train, y_train)

  selected_models['Neural Network'] = mlp_grid_search.best_estimator_

  joblib.dump(mlp_grid_search.best_estimator_, '../models/neural_network.pkl')

  print('Best score: {}'.format(mlp_grid_search.best_score_))
  print('Best parameters: {}'.format(mlp_grid_search.best_params_))
  print('Training time: {}'.format(mlp_grid_search.refit_time_))
  display_mlp_loss_curve(selected_models['Neural Network'])
  display_mlp_weights(selected_models['Neural Network'])
```


```python
if load_models:
    selected_models['Neural Network'] = joblib.load('../models/neural_network.pkl')
    display_mlp_loss_curve(selected_models['Neural Network'])
    display_mlp_weights(selected_models['Neural Network'])
```


    
![png](../images/output_75_0.png)
    



    
![png](../images/output_75_1.png)
    


#### Gaussian Naive Bayes


```python
if not load_models:

    parameter_grid = {'var_smoothing': [1e-9, 1e-10, 1e-11, 1e-12, 1e-13]}


    gnb_grid_search = GridSearchCV(estimator=naive_bayes_classifier,
                                param_grid=parameter_grid,
                                cv=cross_validation,
                                n_jobs=-1)

    gnb_grid_search.fit(X_train, y_train)

    selected_models['Naive Bayes'] = gnb_grid_search.best_estimator_

    joblib.dump(gnb_grid_search.best_estimator_, '../models/naive_bayes.pkl')

    print('Best score: {}'.format(gnb_grid_search.best_score_))
    print('Best parameters: {}'.format(gnb_grid_search.best_params_))
    print('Training time: {}'.format(gnb_grid_search.refit_time_))
```


```python
if load_models:
    selected_models['Naive Bayes'] = joblib.load('../models/naive_bayes.pkl')
```

#### Random Forest


```python
if not load_models:

    parameter_grid = {'n_estimators': range(10, 20),
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': range(1, 23),
                    'min_samples_split': range(2, 10),
                    'min_samples_leaf': range(1, 5),
                    'ccp_alpha': [0.0, 0.1, 0.2],
                    'random_state': [random_state]}
                    
    rf_grid_search = GridSearchCV(estimator=random_forest_classifier,
                                param_grid=parameter_grid,
                                cv=cross_validation,
                                n_jobs=-1)

    rf_grid_search.fit(X_train, y_train)

    selected_models['Random Forest'] = rf_grid_search.best_estimator_

    joblib.dump(rf_grid_search.best_estimator_, '../models/random_forest.pkl')

    print('Best score: {}'.format(rf_grid_search.best_score_))
    print('Best parameters: {}'.format(rf_grid_search.best_params_))
    print('Training time: {}'.format(rf_grid_search.refit_time_))
                    
```


```python
if load_models:
    selected_models['Random Forest'] = joblib.load('../models/random_forest.pkl')
```

#### Gradient Boosting


```python
if not load_models:

  parameter_grid = {'loss': ['log_loss', 'exponential'],
                    'learning_rate': [0.01, 0.1, 1.0],
                      'n_estimators': range(10, 20),
                      'criterion' : ['friedman_mse', 'squared_error'],
                      'max_depth': range(1, 23),
                      'min_samples_split': range(2, 10),
                      'min_samples_leaf': range(1, 5),
                      'ccp_alpha': [0.0, 0.1, 0.2],
                      'random_state': [random_state]
  }

  gb_grid_search = GridSearchCV(estimator=gradient_boosting_classifier,
                              param_grid=parameter_grid,
                              cv=cross_validation,
                              n_jobs=-1)

  gb_grid_search.fit(X_train, y_train)

  selected_models['Gradient Boosting'] = gb_grid_search.best_estimator_

  joblib.dump(gb_grid_search.best_estimator_, '../models/gradient_boosting.pkl')

  print('Best score: {}'.format(gb_grid_search.best_score_))
  print('Best parameters: {}'.format(gb_grid_search.best_params_))
  print('Training time: {}'.format(gb_grid_search.refit_time_))
```


```python
if load_models:
    selected_models['Gradient Boosting'] = joblib.load('../models/gradient_boosting.pkl')
```

### Learning Curves

Here, we will plot the learning curves for the best model of each algorithm.

Each learning curve depicts the average evolution of both the training and validation scores throughout the training process of each cross-validation fold.


```python
from sklearn.model_selection import learning_curve

fig = plt.figure(figsize=(12, 10))

for model_name, model in selected_models.items():
    train_sizes, train_scores, validation_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=cross_validation, n_jobs=-1, random_state=random_state)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    
    ax = fig.add_subplot(3, 3, list(selected_models.keys()).index(model_name) + 1)
    fig.tight_layout(pad=4) 
    ax.plot(train_sizes, train_scores_mean, label='Training score')
    ax.plot(train_sizes, validation_scores_mean, label='Cross-validation score')
    ax.set_title(model_name)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.legend()

plt.show()
```


    
![png](../images/output_87_0.png)
    


Most learning curves depict a decreasing trend in the training score. This is likely due to two main factors:
- As the model incorporates more training examples, the overall complexity and diversity of training data increases, which makes it harder for the model to fit the training data perfectly.
- In cross-validation, the models are selected based on the validation score, which means that the models are selected based on their ability to generalize to unseen data. This can lead to a decrease in the training score, as the models avoid overfitting to the training data.

All learning curves depict an increasing trend in the validation score. This is a good indicator that the models are learning the underlying patterns in the dataset and generalizing well.

However, the Naive Bayes model exhibits a concurrent increase in both the training and validation scores. This is likely due to the independence assumption of the Naive Bayes model, which simplifies the model and reduces the risk of overfitting, and its probabilistic nature, which leads to an increase in both the training and validation scores, as the model trains on more data and the probability estimates become more accurate.

### Model Comparison

We wanted to compare the performance of the best model for each algorithm, based on the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Prediction Time

This analysis would allow us to determine which model is the most suitable for the classification task at hand.

The chosen metrics will give us a good understanding of the model's performance, as they provide a comprehensive view of the model's ability to correctly classify the data, as well as the time it takes to make predictions, which is also an important factor to consider, specially in real-time applications (such as medical diagnosis).


```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import timeit

evaluation_parameters = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', "True Positive", "False Positive", "True Negative", "False Negative", "Test Size", "Total Prediction Time (ms)", "Average Prediction Time (ms)"]
model_evaluation = pd.DataFrame(columns=evaluation_parameters)
positive_value = 1
positive_label = 'GBM'
```


```python
test_size = len(y_test)

for model_name, model in selected_models.items():
    start_time = timeit.default_timer()
    y_predicted = model.predict(X_test)
    end_time = timeit.default_timer()
    prediction_time = (end_time - start_time) * 1000
    model_classification_report = classification_report(y_test, y_predicted, labels=[positive_value], target_names=[positive_label], digits=3, ../images/output_dict=True)
    model_confusion_matrix = confusion_matrix(y_test, y_predicted)
    model_evaluation = pd.concat(
        [model_evaluation,
            pd.DataFrame([[model_name,
                            accuracy_score(y_test, model.predict(X_test)),
                            model_classification_report[positive_label]['precision'],
                            model_classification_report[positive_label]['recall'],
                            model_classification_report[positive_label]['f1-score'],
                            model_confusion_matrix[1][1],
                            model_confusion_matrix[0][1],
                            model_confusion_matrix[1][0],
                            model_confusion_matrix[0][0],
                            len(y_test),
                            prediction_time,
                            prediction_time/test_size,
                        ]], columns=evaluation_parameters)],
        ignore_index=True)
```

    C:\Users\tfili\AppData\Local\Temp\ipykernel_21984\3913487951.py:10: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      model_evaluation = pd.concat(
    


```python
model_evaluation.head(len(selected_models))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>True Positive</th>
      <th>False Positive</th>
      <th>True Negative</th>
      <th>False Negative</th>
      <th>Test Size</th>
      <th>Total Prediction Time (ms)</th>
      <th>Average Prediction Time (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nearest Neighbors</td>
      <td>0.827381</td>
      <td>0.804598</td>
      <td>0.853659</td>
      <td>0.828402</td>
      <td>70</td>
      <td>17</td>
      <td>12</td>
      <td>69</td>
      <td>168</td>
      <td>10.2181</td>
      <td>0.060822</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>0.833333</td>
      <td>0.806818</td>
      <td>0.865854</td>
      <td>0.835294</td>
      <td>71</td>
      <td>17</td>
      <td>11</td>
      <td>69</td>
      <td>168</td>
      <td>0.3375</td>
      <td>0.002009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Support Vector</td>
      <td>0.845238</td>
      <td>0.811111</td>
      <td>0.890244</td>
      <td>0.848837</td>
      <td>73</td>
      <td>17</td>
      <td>9</td>
      <td>69</td>
      <td>168</td>
      <td>1.0869</td>
      <td>0.006470</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Neural Network</td>
      <td>0.839286</td>
      <td>0.823529</td>
      <td>0.853659</td>
      <td>0.838323</td>
      <td>70</td>
      <td>15</td>
      <td>12</td>
      <td>71</td>
      <td>168</td>
      <td>0.3352</td>
      <td>0.001995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>0.797619</td>
      <td>0.726415</td>
      <td>0.939024</td>
      <td>0.819149</td>
      <td>77</td>
      <td>29</td>
      <td>5</td>
      <td>57</td>
      <td>168</td>
      <td>0.2786</td>
      <td>0.001658</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest</td>
      <td>0.815476</td>
      <td>0.800000</td>
      <td>0.829268</td>
      <td>0.814371</td>
      <td>68</td>
      <td>17</td>
      <td>14</td>
      <td>69</td>
      <td>168</td>
      <td>0.8863</td>
      <td>0.005276</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gradient Boosting</td>
      <td>0.839286</td>
      <td>0.816092</td>
      <td>0.865854</td>
      <td>0.840237</td>
      <td>71</td>
      <td>16</td>
      <td>11</td>
      <td>70</td>
      <td>168</td>
      <td>0.4225</td>
      <td>0.002515</td>
    </tr>
  </tbody>
</table>
</div>



### Confusion Matrix

Plotting the confusion matrix for each model will provide us a more detailed insight over the model's performance, as it will allow us to see the number of true positives, true negatives, false positives, and false negatives.


```python
from sklearn.metrics import ConfusionMatrixDisplay

fig = plt.figure(figsize=(10, 10))

for model_name, model in selected_models.items():
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, model.predict(X_test)), display_labels=['LGG', 'GBM'])
    confusion_matrix_display.plot(cmap='Blues', ax=fig.add_subplot(3, 3, list(selected_models.keys()).index(model_name) + 1), xticks_rotation='horizontal')
    fig.tight_layout(w_pad=4.0, h_pad=3.0)
    plt.title(model_name)
    
plt.show()
```


    
![png](../images/output_96_0.png)
    


These confusion matrices will be used to further evaluate the performance of the models, along with the information provided by the classification reports.

### Classification Report

Visualizing the classification report for each model in bar plot form will facilitate the comparison of the performance of each model across the different metrics.


```python
import seaborn as sb
import matplotlib.pyplot as plt

model_evaluation_melted = model_evaluation[["Model", "Accuracy", "Precision", "Recall", "F1-score"]].melt(id_vars=["Model"], var_name="Measure", value_name="Value")

sb.barplot(data=model_evaluation_melted, x='Model', y='Value', hue='Measure', palette='viridis')
plt.title('Model Evaluation Metrics')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tick_params(axis='x', labelrotation=270)
plt.yticks(np.arange(0, 1.0, 0.05), minor=True)
plt.show()
```


    
![png](../images/output_100_0.png)
    


By looking at the classification report for each model, we can see that the **Support Vector Machine** and **Neural Network** models have the **highest accuracy**. The **Support Vector Machine** model also has the **highest F1-score**, close to the **Gradient Boosting** model, while the **Neural Network** model has the **highest precision**. As for the **recall**, the **Naive Bayes** model is, by far, the best.

Stating that the **Neural Network** model is the more precise means that it has the lowest number of **false positives**. On the other hand, the **Naive Bayes** model has the highest recall, which means that it has the lowest number of **false negatives**. Since this is a medical diagnosis task, it is specially important to minimize the number of false negatives, as it is crucial to correctly identify the patients with the most severe condition.

The **Support Vector Machine** and **Gradient Boosting** models seem to have the best balance between precision and recall, as they have the highest **F1-score**.

As they have the highest accuracy, which is the most general and widely-used measure of performance, the **Support Vector Machine** and **Neural Network** models could be regarded as the best models for this classification task. Since this metric can be misleading, we will also consider other metrics to accurately evaluate the models.

### Prediction Time

As this is a crucial metric for real-time applications, we will plot the prediction time for each model. Following the context of medical diagnosis, it is important to have a model that can make predictions in a reasonable amount of time.


```python
sb.barplot(data=model_evaluation, x='Model', y='Average Prediction Time (ms)')
plt.title('Prediction Time by Model')
plt.tick_params(axis='x', labelrotation=270)
plt.show();
```


    
![png](../images/output_104_0.png)
    


The **Nearest Neighbors** model takes two orders of magnitude more time to make predictions than most of the other models, while the **Naive Bayes** and **Decision Tree** models are the fastest. Since none of the models takes more than a tenth of a millisecond to make predictions, the prediction time is not a critical factor for a dataset with similar volume. However, considering this task could be scaled to a (much) larger dataset, the prediction time for the **Nearest Neighbors** model could become an issue, as it is significantly higher than the other models. 

### ROC Curve

Illustrating the ROC curve (receiver operating characteristic curve) for each model will allow us to understand the trade-off between the true positive rate and the false positive rate, as well as the area under the curve (AUC), which is a good indicator of the model's performance.


```python
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

fig = plt.figure(figsize=(13, 15))

for model_name, model in selected_models.items():
    y_predicted = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_predicted)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_display.plot(ax=fig.add_subplot(3, 3, list(selected_models.keys()).index(model_name) + 1))
    fig.tight_layout(w_pad=4.0)
    plt.title(model_name)

plt.show()
```


    
![png](../images/output_108_0.png)
    


Since the ROC curve measures the trade-off between the true positive rate and the false positive rate, it provides a comprehensive view of the model's ability to correctly classify the data. Therefore, a higher AUC indicates better performance (i.e., if the AUC was **1**, the model's predictions would be **100%** correct).

For this task, the model with the highest AUC is the **Support Vector Machine** model, followed by the **Neural Network** and the **Gradient Boosting** models. On the other hand, the two models that stand out as the worst are the **Naive Bayes** and the **Random Forest** models. This information is a valuable indicator of which model is the most suitable for the classification task at hand, as we should consider to prioritize the models with the highest AUC.

### Cost Analysis

To properly evaluate and compare the different models, we will calculate the cost for each one. This metric will allow us to penalize the models that have a higher number of false negatives, which is crucial for this medical diagnosis task. For the specific context of brain glioma grading, it is also important to minimize the number of false positives. A patient diagnosed with a false positive could be subjected to unnecessary treatments, which could have a negative impact on their health.


```python
true_positive_cost = -1
false_positive_cost = 19
true_negative_cost = 0
false_negative_cost = 80

model_evaluation['Total Cost'] = (model_evaluation['True Positive'] * true_positive_cost) + (model_evaluation['False Positive'] * false_positive_cost) + (model_evaluation['True Negative'] * true_negative_cost) + (model_evaluation['False Negative'] * false_negative_cost)

model_evaluation.head(len(selected_models))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>True Positive</th>
      <th>False Positive</th>
      <th>True Negative</th>
      <th>False Negative</th>
      <th>Test Size</th>
      <th>Total Prediction Time (ms)</th>
      <th>Average Prediction Time (ms)</th>
      <th>Total Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nearest Neighbors</td>
      <td>0.827381</td>
      <td>0.804598</td>
      <td>0.853659</td>
      <td>0.828402</td>
      <td>70</td>
      <td>17</td>
      <td>12</td>
      <td>69</td>
      <td>168</td>
      <td>10.2181</td>
      <td>0.060822</td>
      <td>5773</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>0.833333</td>
      <td>0.806818</td>
      <td>0.865854</td>
      <td>0.835294</td>
      <td>71</td>
      <td>17</td>
      <td>11</td>
      <td>69</td>
      <td>168</td>
      <td>0.3375</td>
      <td>0.002009</td>
      <td>5772</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Support Vector</td>
      <td>0.845238</td>
      <td>0.811111</td>
      <td>0.890244</td>
      <td>0.848837</td>
      <td>73</td>
      <td>17</td>
      <td>9</td>
      <td>69</td>
      <td>168</td>
      <td>1.0869</td>
      <td>0.006470</td>
      <td>5770</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Neural Network</td>
      <td>0.839286</td>
      <td>0.823529</td>
      <td>0.853659</td>
      <td>0.838323</td>
      <td>70</td>
      <td>15</td>
      <td>12</td>
      <td>71</td>
      <td>168</td>
      <td>0.3352</td>
      <td>0.001995</td>
      <td>5895</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>0.797619</td>
      <td>0.726415</td>
      <td>0.939024</td>
      <td>0.819149</td>
      <td>77</td>
      <td>29</td>
      <td>5</td>
      <td>57</td>
      <td>168</td>
      <td>0.2786</td>
      <td>0.001658</td>
      <td>5034</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest</td>
      <td>0.815476</td>
      <td>0.800000</td>
      <td>0.829268</td>
      <td>0.814371</td>
      <td>68</td>
      <td>17</td>
      <td>14</td>
      <td>69</td>
      <td>168</td>
      <td>0.8863</td>
      <td>0.005276</td>
      <td>5775</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gradient Boosting</td>
      <td>0.839286</td>
      <td>0.816092</td>
      <td>0.865854</td>
      <td>0.840237</td>
      <td>71</td>
      <td>16</td>
      <td>11</td>
      <td>70</td>
      <td>168</td>
      <td>0.4225</td>
      <td>0.002515</td>
      <td>5833</td>
    </tr>
  </tbody>
</table>
</div>



We could see that the **Naive Bayes** model has the lowest cost, as it has the lowest number of false negatives. On the other hand, the **Neural Network** model has the highest cost, as it has the highest number of false negatives.


```python
model_evaluation['Accuracy/Cost Ratio'] = model_evaluation['Accuracy'] / model_evaluation['Total Cost']

model_evaluation.head(len(selected_models))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>True Positive</th>
      <th>False Positive</th>
      <th>True Negative</th>
      <th>False Negative</th>
      <th>Test Size</th>
      <th>Total Prediction Time (ms)</th>
      <th>Average Prediction Time (ms)</th>
      <th>Total Cost</th>
      <th>Accuracy/Cost Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nearest Neighbors</td>
      <td>0.827381</td>
      <td>0.804598</td>
      <td>0.853659</td>
      <td>0.828402</td>
      <td>70</td>
      <td>17</td>
      <td>12</td>
      <td>69</td>
      <td>168</td>
      <td>10.2181</td>
      <td>0.060822</td>
      <td>5773</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>0.833333</td>
      <td>0.806818</td>
      <td>0.865854</td>
      <td>0.835294</td>
      <td>71</td>
      <td>17</td>
      <td>11</td>
      <td>69</td>
      <td>168</td>
      <td>0.3375</td>
      <td>0.002009</td>
      <td>5772</td>
      <td>0.000144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Support Vector</td>
      <td>0.845238</td>
      <td>0.811111</td>
      <td>0.890244</td>
      <td>0.848837</td>
      <td>73</td>
      <td>17</td>
      <td>9</td>
      <td>69</td>
      <td>168</td>
      <td>1.0869</td>
      <td>0.006470</td>
      <td>5770</td>
      <td>0.000146</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Neural Network</td>
      <td>0.839286</td>
      <td>0.823529</td>
      <td>0.853659</td>
      <td>0.838323</td>
      <td>70</td>
      <td>15</td>
      <td>12</td>
      <td>71</td>
      <td>168</td>
      <td>0.3352</td>
      <td>0.001995</td>
      <td>5895</td>
      <td>0.000142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>0.797619</td>
      <td>0.726415</td>
      <td>0.939024</td>
      <td>0.819149</td>
      <td>77</td>
      <td>29</td>
      <td>5</td>
      <td>57</td>
      <td>168</td>
      <td>0.2786</td>
      <td>0.001658</td>
      <td>5034</td>
      <td>0.000158</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest</td>
      <td>0.815476</td>
      <td>0.800000</td>
      <td>0.829268</td>
      <td>0.814371</td>
      <td>68</td>
      <td>17</td>
      <td>14</td>
      <td>69</td>
      <td>168</td>
      <td>0.8863</td>
      <td>0.005276</td>
      <td>5775</td>
      <td>0.000141</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gradient Boosting</td>
      <td>0.839286</td>
      <td>0.816092</td>
      <td>0.865854</td>
      <td>0.840237</td>
      <td>71</td>
      <td>16</td>
      <td>11</td>
      <td>70</td>
      <td>168</td>
      <td>0.4225</td>
      <td>0.002515</td>
      <td>5833</td>
      <td>0.000144</td>
    </tr>
  </tbody>
</table>
</div>



By analysing the accuracy/cost ratio for each model, we could see that the **Naive Bayes** model stood out as the best, as it has the highest ratio. If the goal of the classification task is to really enphasize the importance of minimizing the number of false negatives, this model would probably be the best choice. It also has the advantage of being the fastest model to make predictions. 

### Exporting the models results to a file


```python
model_evaluation.to_csv('../out/glioma_model_evaluation.csv', index=False)
```

## Conclusion

Regarding the data pre-processing phase of the project, we concluded that it allowed us to simplify the original dataset and merely keep relevant data. 
Furthermore, we were able to treat and convert the data to a more suitable format, which proved to be an essential step in the supervised learning process.
Additionally, the pre-analysis of the dataset gave us some valuable insights that denoted parallels with the conclusions drawn later from the results obtained after the application of the supervised learning techniques.

The grid search approach, with stratified k-fold cross-validation, proved to be a valuable tool in the hyperparameter tuning process. 
This technique allowed us to find the optimal hyperparameters for the selected models, which resulted in the best possible performance for each of them. 
The stratified k-fold cross-validation also ensured that the models were trained and tested on balanced data, which is fundamental to guarantee the models' generalization to unseen data.

With respect to the evaluation of the different supervised learning models, we concluded that among the obtained models, the Support Vector Machine, Neural Network and Gradient Boosting models had the overall best performance.
The grid-search process and finetuning for the Support Vector Machine model was, undeniably, the most time and resource efficient, and this model also presented the best trade off between performance and computational cost.
On the other hand, the Nearest Neighbor model performed the worst, with the lowest accuracy and, by far, the highest prediction time.

The Decision Tree model performed well too, and showed interesting results regarding each feature's impact on the classification task. 
This was the only model that could provide a clear insight into the importance of the attributes, facilitating its interpretation and understanding.
Furthermore, the features chosen by this model aligned with the pre-analysis of the dataset, since they were congruent with the features that showed the highest correlation to the target variable.

Considering an analysis of the selected models that is more sensitive to the context of the classification task at hand, we also computed a weighted cost of the entries of the confusion matrices for each model. In this analysis, the Gaussian Naive Bayes model performed the best, with the lowest weighted cost, due to its lower false negative rate.
On the contrary, the Neural Network model performed the worst, making it the least suitable model for the chosen cost evaluation criteria.

In conclusion, stating that a model is the best suited for a given classification task is a complex and multifaceted statement. This statement really depends on the criteria that are most important for the problem at hand. Since the Glioma grading task corresponds to a medical diagnosis, we determined that the F1-score measure was the best indicator of the models performance. This metric is a good balance between precision and recall, which are both crucial for this task.

## Extra Section

### Dimensionality Reduction

In this section, we will try to find the optimal subset of mutation genes and clinical features for the glioma grading process to improve/maintain performance and reduce costs. The goal is to reduce the dimensionality of the dataset, allowing for faster training and prediction times, while still maintaining a good level of classification performance.


```python
sb.heatmap(data.corr()[['Grade']].drop('Grade').T, annot=False, cmap='seismic', center=0);
plt.title('Correlation between features and target');
```


    
![png](../images/output_123_0.png)
    


By the analysis of the correlation between the features in the dataset and the target variable, we could see that probably not all the features are equally relevant for the classification task. In order to reduce the dimensionality of the dataset, we will try to remove some of the features that are less relevant.

Even though some features have a low correlation with a target variable, they still can play a role in the classification task, as they can be relevant when combined with other features. As we already removed the redundant columns in the data pre-processing phase, we will use the **Recursive Feature Elimination** (RFE) method to rank the features according to their importance and select the best subset of features for the classification task.

As the **Support Vector Machine** model proved to have the best performance, we will use it has the estimator for the RFE method.


```python
from sklearn.feature_selection import RFECV

data_simplified = data.copy()

rfecv = RFECV(estimator=selected_models["Support Vector"], cv=cross_validation, scoring='accuracy', n_jobs=-1, verbose=1)

rfecv.fit(X_train, y_train)
rfecv_score = rfecv.score(X_test, y_test)

start_time = timeit.default_timer()
rfecv.predict(X_test)
end_time = timeit.default_timer()

avg_prediction_time = (end_time - start_time) * 1000 / len(X_test)

print(f"\n- Optimal number of features: {rfecv.n_features_}")
print(f"- Score: {round(rfecv_score * 100, 3)}%")
print(f"- Average prediction time: {round(avg_prediction_time, 3)} ms")

print("\nComparing the model with the selected features:")

accuracy_diff = rfecv_score - model_evaluation[model_evaluation['Model'] == 'Support Vector']['Accuracy'].values[0]
accuracy_ratio = accuracy_diff / model_evaluation[model_evaluation['Model'] == 'Support Vector']['Accuracy'].values[0]
prediction_time_ratio = avg_prediction_time / model_evaluation[model_evaluation['Model'] == 'Support Vector']['Average Prediction Time (ms)'].values[0] 

print(f"\n- Accuracy: {round(accuracy_ratio, 3)}% worse")
print(f"\n- Prediction time: {round(prediction_time_ratio * 100, 3)}% faster")

data_simplified = data_simplified.drop(columns=data.columns[1:][~rfecv.support_])

sb.heatmap(data_simplified.corr()[['Grade']].drop('Grade').T, annot=False, cmap='seismic', center=0);
plt.title('Correlation between features and target');
```

    Fitting estimator with 23 features.
    Fitting estimator with 22 features.
    Fitting estimator with 21 features.
    Fitting estimator with 20 features.
    Fitting estimator with 19 features.
    Fitting estimator with 18 features.
    Fitting estimator with 17 features.
    Fitting estimator with 16 features.
    Fitting estimator with 15 features.
    Fitting estimator with 14 features.
    Fitting estimator with 13 features.
    Fitting estimator with 12 features.
    Fitting estimator with 11 features.
    Fitting estimator with 10 features.
    
    - Optimal number of features: 9
    - Score: 84.524%
    - Average prediction time: 0.005 ms
    
    Comparing the model with the selected features:
    
    - Accuracy: 0.0% worse
    
    - Prediction time: 75.554% faster
    


    
![png](../images/output_125_1.png)
    


By comparing the performance of the model with all the features and the model with the selected subset of features, we determined that the dimensionality reduction was beneficial, as the model with the selected subset of features had a similar performance to the model with all the features, while taking less time to make predictions.

On an additional note, it is important to mention that the optimal number of features and the subset of selected features would be different for each model, so this subset can only be considered optimal for the **Support Vector Machine** model.

### Exporting the simplified dataset file


```python
data_simplified.to_csv("../data/TCGA_GBM_LGG_Mutations_simplified.csv", index=True)
```
