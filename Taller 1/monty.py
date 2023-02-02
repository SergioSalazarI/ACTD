from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork ([( "C" , "A" ) , ( "U" , "A" ) ])

cpd_c = TabularCPD(variable="C", variable_card=3, values=[[0.33], [0.33], [0.33]])
cpd_u = TabularCPD(variable="U", variable_card=3, values=[[0.33], [0.33], [0.33]])

cpd_a = TabularCPD(
    variable="A",
    variable_card=3,
    values=[
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]
    ],
    evidence=["C", "U"],
    evidence_card=[3, 3],
)

model.add_cpds(cpd_c, cpd_u, cpd_a)

model.check_model()

infer = VariableElimination(model)

posterior_p = infer.query(["C"], evidence={"U": 1, "A": 2})
#posterior_p = infer.query(["C"], evidence={"U": 1})
#posterior_p = infer.query(["C"], evidence={"A": 1})
print(posterior_p)

print("A continuación se imprimen las independencias del modelo.")
print( model.get_independencies())