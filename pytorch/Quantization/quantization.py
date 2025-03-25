import numpy as np
# Suppress scientific notation
np.random.state=79
np.set_printoptions(suppress=True)
params=np.random.uniform(low=-50,high=150,size=20)

params[0]=params.max()+1
params[1]=params.min()-1
params[2]=0

params=np.round(params,2)
print(params)

def clamp(params_q:np.array,lower_bound:int,upper_bound:int)->np.array:
    params_q[params_q<lower_bound]=lower_bound
    params_q[params_q>upper_bound]=upper_bound
    return params_q

def asymmetric_quantization(params:np.array,bits:int)->tuple[np.array,float,int]:
    alpha=np.max(params)
    beta=np.min(params)
    scale=(alpha-beta)/(2**bits-1)
    zero=-1*np.round(beta/scale)
    lower_bound,upper_bound=0,2**bits-1
    quantized=clamp(np.round(params / scale + zero),lower_bound,upper_bound).astype(np.int32)
    return quantized,scale,zero

def asymmetric_dequantize(params_q:np.array,scale:float,zero:int)->np.array:
    return (params_q-zero)*scale

def symmetric_quantization(params:np.array,bits:int)->np.array:
    alpha=np.max(params)
    beta=np.min(params)
    scale=(alpha-beta)/(2**bits-1)
    lower_bound=-2**(bits-1)
    upper_bound=2**(bits-1)
    quantized=clamp(np.round(params / scale ),lower_bound,upper_bound).astype(np.int32)
    return quantized,scale
def symmetric_dequantize(params:np.array,scale:float)->np.array:
    return params*scale

def quantization_error(params:np.array,params_q:np.array):
    return np.mean((params-params_q)**2)
(asymmetric_q, asymmetric_scale, asymmetric_zero) = asymmetric_quantization(params, 8)
(symmetric_q, symmetric_scale) = symmetric_quantization(params, 8)

print(f'Original:')
print(np.round(params, 2))
print('')
print(f'Asymmetric scale: {asymmetric_scale}, zero: {asymmetric_zero}')
print(asymmetric_q)
print('')
print(f'Symmetric scale: {symmetric_scale}')
print(symmetric_q)
# Dequantize the parameters back to 32 bits
params_deq_asymmetric = asymmetric_dequantize(asymmetric_q, asymmetric_scale, asymmetric_zero)
params_deq_symmetric = symmetric_dequantize(symmetric_q, symmetric_scale)
# Calculate the quantization error
print(f'{"Asymmetric error: ":>20}{np.round(quantization_error(params, params_deq_asymmetric), 2)}')
print(f'{"Symmetric error: ":>20}{np.round(quantization_error(params, params_deq_symmetric), 2)}')



