# Consolidating Employee Data

The mission is to structure the human resources data at a small
business experiencing exponential growth. The data is currently scattered
across teams and files and comes in various formats: Excel files, CSVs,
JSON files...

the following is format of data in the `datasets` folder:

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







