import argparse
import json
from prepare import ( 
    build_pytorch_docker_enviroment 
    )


parser = argparse.ArgumentParser()
parser.add_argument("--container", type=str, help="The docker container to use, if not exists the script will create one", default='guilty_commit', required=False)
parser.add_argument("--build", type=str, help="The build method, source or nightly or existing", default="existing", required=False)
parser.add_argument("--download", action="store_true", default=False, help="If set, download pytorch source code.", required=False)

args = parser.parse_args()


enviroments = build_pytorch_docker_enviroment(args.download, args.build, args.container)

with open(f"results/enviroments.log", 'w') as f:
    f.write(f"{enviroments}\n")

if len(enviroments) == 0:
    print("Failed to build pytorch enviroment.")
    exit(1)

