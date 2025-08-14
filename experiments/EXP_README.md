## Using docker
To run in the server
```bash
docker build -t image_name .
```
```bash
docker run -d -v /home/danlawand/.gurobi/gurobi.lic:/opt/gurobi/gurobi.lic:ro -v /home/danlawand/causal-reasoning/resultados:/code/outputs image_name python3 experiment.py
```