set -e

rm -rf dist
rm -rf build

rm -rf sam2.1_hiera_large.pt
./src/sam2/checkpoints/download_ckpts.sh

mv sam2.1_hiera_large.pt src/sam2/checkpoints/

uv build --wheel
twine upload --repository-url http://192.168.1.50:8080/ -u c6-server-01 -p 6SIGMAINT dist/*.whl