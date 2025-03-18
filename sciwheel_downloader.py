#!/usr/bin/env python

"""

Usage:
  sciwheel_downloader.py [-v] [-k <token>] <project> <outdir>


Options:
  -v                       Verbose output

  -k <token>               API Token [Default: os.sys.getenv('SCIWHEEL_TOKEN')]

Arguments:

  <project>                Name of the project to fetch

  <outdir>                 Directory to store downloaded references

"""

import os
import json
import requests
from docopt import docopt
from urllib.parse import urlparse

def get_project_id(api_token, project_name):
    url = "https://sciwheel.com/extapi/work/projects?size=85"
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    total_n = response.json().get("total")
    print(f"Number of projects: {total_n}")
    projects = response.json().get("results", [])
    
    for project in projects:
        if verbose:
            print(f"project: {project['name']}")
        if project["name"] == project_name:
            return project["id"]
    
    raise ValueError(f"Project '{project_name}' not found.")

def get_references(api_token, project_id):
    url = f"https://sciwheel.com/extapi/work/references?projectId={project_id}&size=100"
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("results", [])

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


verbose = True
def main():
    args = docopt(__doc__)
    api_token = args["-k"]
    project_name = args["<project>"]
    output_directory = args["<outdir>"]


    if verbose:
        print(f"token: {api_token}, project: {project_name}, outdir: {output_directory}")
    
    
    os.makedirs(output_directory, exist_ok=True)
    metadata_dir = os.path.join(output_directory, "metadata")
    pdf_dir = os.path.join(output_directory, "pdfs")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    project_id = get_project_id(api_token, project_name)
    references = get_references(api_token, project_id)
    
    existing_files = set(f for f in os.listdir(metadata_dir) if f.endswith(".json"))
    
    for ref in references:
        ref_id = ref["id"]
        metadata_file = os.path.join(metadata_dir, f"{ref_id}.json")
        pdf_file = os.path.join(pdf_dir, f"{ref_id}.pdf")
        
        if f"{ref_id}.json" in existing_files:
            if verbose:
                print(f"Skipping already downloaded reference {ref_id}.")
            continue
        
        with open(metadata_file, 'w') as f:
            json.dump(ref, f, indent=2)
        
        pdf_url = ref.get("pdfUrl")
        if pdf_url:
            try:
                download_file(pdf_url, pdf_file)
                if verbose:
                    print(f"Downloaded PDF for reference {ref_id}.")
            except requests.RequestException:
                if verbose:
                    print(f"Failed to download PDF for reference {ref_id}.")
    
    if verbose:
        print("Download process completed.")

if __name__ == "__main__":
    main()
