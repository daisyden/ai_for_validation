import json
from github_issue import Github_Issue 

repo = "intel/torch-xpu-ops"
token = ""
github_issue = Github_Issue(repo, token)

raw_report = "issue_checker_results_good.txt"
report = "issue_checker_report.csv"

data_list = []

try:
    with open(raw_report, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) != 0:
                print(line)
                data = json.loads(line)
                data_list.append(data)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

bug_scrub_owner="chuanqi129"
mailing = []

with open(report, "w") as file:
    for item in data_list:
        reporter_tbd = [] 
        scrub_tbd = [] 
        owner_tbd = [] 

        for key in item:
            if key in ["issue_desscription", "error_message", "impact"]:
                if item[key]['score'] == 0:
                    reporter_tbd.append(key)
    
            elif key in ["reproduce_steps"]:
                for _key in [ "steps", "platform", "software_version" ]:
                    if item[key][_key]['score'] == 0:
                        reporter_tbd.append(f"reproduce_{_key}") 
    
            elif key == "predicted_module":
                if item['predicted_module']['module'] != "N/A" and item['labeled_module']['module'] != item['predicted_module']['module']:
                    scrub_tbd.append(f"module:{item['predicted_module']['module']}")
    
            elif key == "assignee" and item['assignee'] == "":
                scrub_tbd.append("owner") 
    
            elif key == "last_update":
                from datetime import datetime
                
                def date_delta(date_str, date_format):
                    """
                    Calculates the difference in days between a given date string and today's date.
                
                    Args:
                        date_str (str): The date string to compare, e.g., "2025-05-15".
                        date_format (str): The format of the date string, e.g., "%Y-%m-%d".
                
                    Returns:
                        int: The difference in days, positive if the date is in the future,
                             negative if in the past, and 0 if it's today.
                    """
                    today = datetime.now()
                    try:
                        input_date = datetime.strptime(date_str, date_format)
                    except ValueError:
                        raise ValueError("Invalid date format. Please use the correct format.")
                
                    delta = input_date - today
                    return delta.days
    
                date_string = item['last_update']['update'] 
                format_string = "%Y-%m-%d"
                try:
                    days_diff = date_delta(date_string, format_string)
                except ValueError as e:
                  print(f"Error: {e}")
    
                # 2 weeks
                if days_diff < -14:
                    owner_tbd.append(f"{item['last_update']['update']}")
        
        issue_tbd = f"{item['issue_number']}"
        _reporter_tbd = ""
        _owner_tbd = ""
        _scrub_tbd = ""
        if len(reporter_tbd) != 0 :
            tbd = ", ".join(reporter_tbd)
            _reporter_tbd = f"@{item['reporter']}, please help to improve {tbd}"
            if item['reporter'] not in mailing:
                user = github_issue.get_user(item['reporter'])
                if user.email != None:
                    mailing.append(user.email)
                else:
                    print("Cannot find email for {}".format(user))
    
    
        if len(owner_tbd) != 0:
            tbd = ", ".join(owner_tbd)
            if item['assignee'] != "":
                _owner_tbd = f"@{item['assignee']}, please help to follow up the issue as it is only updated on {tbd}"
            else:
                _owner_tbd = f"@{bug_scrub_owner}, please help to assign an owner to follow up the issue as it is only updated on {tbd}"
    
            if item['assignee'] not in mailing:
                if item['assignee'] != "":
                    user = github_issue.get_user(item['assignee'])
                    if user.email != None:
                        mailing.append(user.email)
                    else:
                        print("Cannot find email for {}".format(user))
    
    
        if len(scrub_tbd) != 0:
            tbd = ", ".join(scrub_tbd)
            _scrub_tbd = f"@{bug_scrub_owner}, please help to set {tbd}"
            if bug_scrub_owner not in mailing:
                user = github_issue.get_user(bug_scrub_owner)
                if user.email != None:
                    mailing.append(user.email)
                else:
                    print("Cannot find email for {}".format(user))
    
    
        if len(reporter_tbd) != 0  or len(owner_tbd) != 0  or len(scrub_tbd) != 0:
            print(f"https://github.com/intel/torch-xpu-ops/issues/{issue_tbd} | {_reporter_tbd} | {_owner_tbd} | {_scrub_tbd} | {item['issue_type']} | {item['milestone']}\n")
            file.write(f"https://github.com/intel/torch-xpu-ops/issues/{issue_tbd} | {_reporter_tbd} | {_owner_tbd} | {_scrub_tbd} | {item['issue_type']} | {item['milestone']}\n")
    
print("mail to {}".format("; ".join(mailing)))

 

