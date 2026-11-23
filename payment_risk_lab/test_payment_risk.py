from payment_risk_lab.main import simulate,cost,psi
import numpy as np

def test_reproducible():assert simulate(2000,4).equals(simulate(2000,4))
def test_no_duplicate_transactions():
    d=simulate(5000);assert d.transaction_id.is_unique and d.day.between(0,119).all()
def test_asymmetric_cost():
    assert cost([1,0,1],[.9,.8,.2],.5)==(4+4+150)/3
    assert cost([1,0,1],[.9,.8,.2],1.0)==300/3
def test_psi_shift():
    a=np.random.default_rng(1).normal(size=4000)
    assert psi(a,a)==0 and psi(a,a+1)>.1
