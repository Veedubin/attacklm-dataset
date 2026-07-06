import os
import yaml
from collections import OrderedDict

# Custom YAML loader to preserve the order of keys
class OrderedLoader(yaml.SafeLoader):
    pass

def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))

OrderedLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_mapping
)

# Custom YAML dumper to preserve the order of keys
class OrderedDumper(yaml.SafeDumper):
    pass

def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())

OrderedDumper.add_representer(OrderedDict, dict_representer)

# Define the paths
log_file_path = 'data_source_validation.log'
data_sources_dir = 'data_sources'
contentctl_file_path = 'contentctl.yml'

def update_data_sources(ta_name, latest_version):
    # Update the YAML files in the data sources directory
    for filename in os.listdir(data_sources_dir):
        if filename.endswith('.yml'):
            file_path = os.path.join(data_sources_dir, filename)
            with open(file_path, 'r') as yml_file:
                data = yaml.load(yml_file, Loader=OrderedLoader)

            # Check if the TA name matches and update the version
            updated = False
            for ta in data.get('supported_TA', []):
                if ta['name'] == ta_name:
                    if ta['version'] != latest_version:
                        ta['version'] = latest_version
                        updated = True

            # Write the updated data back to the YAML file
            if updated:
                with open(file_path, 'w') as yml_file:
                    yaml.dump(data, yml_file, Dumper=OrderedDumper)

def update_contentctl_yml(title, new_version):
    # Load the existing YAML file
    with open(contentctl_file_path, 'r') as file:
        content = yaml.load(file, Loader=OrderedLoader)

    # Iterate over the apps to find the title and update the version and hardcoded_path
    updated = False
    for app in content.get('apps', []):
        if app.get('title') == title:
            if app.get('version') != new_version:
                app['version'] = new_version
                updated = True
                print(f"Updated {title} in contentctl.yml to version {new_version}")

            # Update the hardcoded_path if it exists
            if 'hardcoded_path' in app:
                base_url, current_version = app['hardcoded_path'].rsplit('_', 1)
                new_hardcoded_path = f"{base_url}_{new_version.replace('.', '')}.tgz"
                app['hardcoded_path'] = new_hardcoded_path
                print(f"Updated hardcoded_path for {title} to {new_hardcoded_path}")

    # Write the updated content back to the YAML file if changes were made
    if updated:
        with open(contentctl_file_path, 'w') as file:
            yaml.dump(content, file, Dumper=OrderedDumper, default_flow_style=False)

def main():
    # Read the log file to find version mismatches
    with open(log_file_path, 'r') as log_file:
        log_lines = log_file.readlines()

    # Parse the log file to find the TA name and the latest version
    for i, line in enumerate(log_lines):
        if 'Version mismatch' in line:
            ta_name = log_lines[i].split("'")[3].strip()
            latest_version = log_lines[i + 1].split(':')[1].strip()
            print(f"Found version mismatch for TA: {ta_name}, updating to version: {latest_version}")

            # Update data sources and contentctl.yml
            update_data_sources(ta_name, latest_version)
            update_contentctl_yml(ta_name, latest_version)

    print("Version updates completed.")

if __name__ == "__main__":
    main()