This project serves as a demonstration of my coding skills and practical application in completing the Data Science Certificate projects provided by DataCamp. 
---
jupyter:
  colab:
    name: Welcome to DataCamp Workspaces.ipynb
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.5
  nbformat: 4
  nbformat_minor: 5
---

<div class="cell markdown">

You just got hired as the first and only data practitioner at a small
business experiencing exponential growth. The company needs more
structured processes, guidelines, and standards. Your first mission is
to structure the human resources data. The data is currently scattered
across teams and files and comes in various formats: Excel files, CSVs,
JSON files...

You'll work with the following data in the `datasets` folder:

-   **Office addresses**
    -   Saved in `office_addresses.csv`.
    -   If the value for office is `NaN`, then the employee is remote.
-   **Employee addresses**
    -   Saved on the first tab of `employee_information.xlsx`.
-   **Employee emergency contacts**
    -   Saved on the second tab of `employee_information.xlsx`; this tab
        is called `emergency_contacts`.
    -   However, this sheet was edited at some point, and ***the headers
        were removed***! The HR manager let you know that they should
        be: `employee_id`, `last_name`, `first_name`,
        `emergency_contact`, `emergency_contact_number`, and
        `relationship`.
-   **Employee roles, teams, and salaries**
    -   This information has been exported from the company's human
        resources management system into a JSON file titled
        `employee_roles.json`. Here are the first few lines of that
        file:

    <!-- -->

        {"A2R5H9":
        {
          "title": "CEO",
          "monthly_salary": "$4500",
          "team": "Leadership"
        },
         ...
        }

</div>

<div class="cell code" execution_count="14" executionTime="23"
id="bA5ajAmk7XH6" lastSuccessfullyExecutedCode="import pandas as pd
# Start coding here... ">

``` python
import pandas as pd
import json
Office_addr = pd.read_csv("office_addresses.csv")
print(Office_addr.head())
Employee_addr = pd.read_excel("employee_information.xlsx", "employee_addresses")
Employee_emrg = pd.read_excel("employee_information.xlsx", "emergency_contacts", names = ["employee_id", "last_name", "first_name", "emergency_contact", "emergency_contact_number","relationship"])

# Opening JSON file
with open("employee_roles.json") as json_file:
    Employee_info = json.load(json_file)

# Converting the json to dataframe
    
Employee_info_df = pd.DataFrame.from_dict(Employee_info, orient="index").reset_index()
Employee_info_df = Employee_info_df.rename(columns={'index':'employee_id' })


# Merging data frames
Employee_addr_emrg = Employee_addr.merge(Employee_emrg, on = "employee_id",  how = "outer")
Employee_addr_emrg = Employee_addr_emrg.drop(['last_name', 'first_name'], axis = 1)


# Merging Employee_ addr_emrg to office info on county
Employee_addr_emrg_office = Employee_addr_emrg.merge(Office_addr, left_on = "employee_country", right_on = "office_country",  how = "outer")

# Merging Employee_addr_emrg_office  to json file data
employees_final = Employee_addr_emrg_office.merge(Employee_info_df, on = 'employee_id', how = "outer")#.reset_index()

# Converting NAN values in office column to "Remote"
employees_final[employees_final["office"].isnull() == True] = "Remote"

# Set employee_id as index

print(employees_final.head())
employees_final.reset_index()
employees_final.set_index("employee_id", inplace=True)



```

<div class="output stream stdout">

              office office_country    office_city   office_street  \
    0  Leuven Office             BE         Leuven  Martelarenlaan   
    1     ESB Office             US  New York City    Fifth Avenue   
    2  WeWork Office             GB         London      Old Street   

       office_street_number  
    0                    38  
    1                   350  
    2                   207  
      employee_id employee_last_name employee_first_name employee_country  \
    0      A2R5H9             Hunman                 Jax               BE   
    1      G4R7V0              Sagal               Gemma               US   
    2      H8K0L6               Siff                Tara               GB   
    3      Remote             Remote              Remote           Remote   

      employee_city employee_street employee_street_number emergency_contact  \
    0        Leuven     Grote Markt                      9               NaN   
    1      New-York    Perry Street                     66      John Newmark   
    2        London    Baker Street                    221   Wendy de Matteo   
    3        Remote          Remote                 Remote            Remote   

      emergency_contact_number relationship         office office_country  \
    0                      NaN          NaN  Leuven Office             BE   
    1           +1-202-555-194      Husband     ESB Office             US   
    2         +44-020-5554-333       Sister  WeWork Office             GB   
    3                   Remote       Remote         Remote         Remote   

         office_city   office_street office_street_number               title  \
    0         Leuven  Martelarenlaan                 38.0                 CEO   
    1  New York City    Fifth Avenue                350.0  Business Developer   
    2         London      Old Street                207.0                 CFO   
    3         Remote          Remote               Remote              Remote   

      monthly_salary        team  
    0          $4500  Leadership  
    1          $3000       Sales  
    2          $4500  Leadership  
    3         Remote      Remote  

</div>

<div class="output stream stderr">

    C:\Users\CSStaff-MS02\AppData\Local\Temp\ipykernel_15564\395789626.py:30: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value 'Remote' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
      employees_final[employees_final["office"].isnull() == True] = "Remote"
    C:\Users\CSStaff-MS02\AppData\Local\Temp\ipykernel_15564\395789626.py:30: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value 'Remote' has dtype incompatible with float64, please explicitly cast to a compatible dtype first.
      employees_final[employees_final["office"].isnull() == True] = "Remote"

</div>

</div>

<div class="cell code" execution_count="6">

``` python
nbconvert --to markdown notebook.ipynb
```

<div class="output error" ename="SyntaxError"
evalue="invalid syntax (479693054.py, line 1)">

      Cell In[6], line 1
        nbconvert --to html notebook.ipynb
                       ^
    SyntaxError: invalid syntax

</div>

</div>

<div class="cell code">

``` python
```

</div>
