from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import itertools

model = BayesianNetwork ([( "R" , "A" ),
                          ( "S" , "A" ),
                          ( "A" , "J" ),
                          ( "A" , "M" )])

# ** CODIFICACION
# 0: verdadero, 1: falso

cpd_r = TabularCPD(variable="R", variable_card=2, values=[[0.01], [0.99]], state_names={"R": ["V", "F"]})
cpd_s = TabularCPD(variable="S", variable_card=2, values=[[0.02], [0.98]], state_names={"S": ["V", "F"]})

cpd_a = TabularCPD(
    variable="A",
    variable_card=2,
    values=[
        [0.95, 0.94, 0.29, 0.001],
        [0.05, 0.06, 0.71, 0.999]
    ],
    evidence=["R", "S"],
    evidence_card=[2, 2],
    state_names={
        "A": ["V", "F"],
        "R": ["V", "F"],
        "S": ["V", "F"]
    }
)

cpd_j = TabularCPD(
    variable="J",
    variable_card=2,
    values= [
        [0.9, 0.05],
        [0.1, 0.95]
    ],
    evidence=["A"],
    evidence_card=[2],
    state_names={
        "J": ["V", "F"],
        "A": ["V", "F"]
    }
)

cpd_m = TabularCPD(
    variable="M",
    variable_card=2,
    values= [
        [0.7, 0.01],
        [0.3, 0.99]
    ],
    evidence=["A"],
    evidence_card=[2],
    state_names={
        "M": ["V", "F"],
        "A": ["V", "F"]
    }
)

model.add_cpds(cpd_r, cpd_s, cpd_a, cpd_j, cpd_m)

model.check_model()

infer = VariableElimination(model)

ss = ["V","F"]

for i,j in itertools.product(ss,ss):
    posterior_p = infer.query(["R"], evidence={"J":i, "M": j})
    print("\n")
    print(f"___ J: {i}  M:{j} ___")
    print(posterior_p)