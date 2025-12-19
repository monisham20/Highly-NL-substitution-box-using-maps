# Highly-NL-substitution-box-using-maps
Generation of highly nonlinear and dynamic s-box using logistic chaotic map:
The project “Generation of Highly Nonlinear and Dynamic S-Box Using Logistic Chaotic Map” focuses on enhancing the security of block ciphers by designing a strong substitution box (S-box) using chaos theory. In symmetric cryptography, the S-box is the only nonlinear component and plays a critical role in resisting cryptanalytic attacks such as linear and differential cryptanalysis. Traditional static S-boxes (like the AES S-box) are fixed and publicly known, which can make them vulnerable to advanced attacks. To overcome this limitation, the proposed approach uses the logistic chaotic map, a simple yet powerful nonlinear mathematical model, to generate dynamic and key-dependent S-boxes with high unpredictability and randomness.

In this method, the logistic map equation 
𝑥
𝑛
+
1
=
𝑟
𝑥
𝑛
(
1
−
𝑥
𝑛
)
x
n+1
	​

=rx
n
	​

(1−x
n
	​

) is used, where the control parameter 
𝑟
r (typically in the range 3.57–4.0) and the initial condition 
𝑥
0
x
0
	​

 act as secret keys. Due to the chaotic nature of the map, even a minute change in these parameters produces entirely different sequences. The generated chaotic sequence is processed through quantization, permutation, and modular operations to construct an 8×8 S-box that satisfies bijectivity. Since the S-box is generated dynamically based on the secret key, each encryption session can use a different S-box, significantly increasing resistance to known-plaintext and chosen-plaintext attacks.

The performance of the proposed chaotic S-box is evaluated using standard cryptographic criteria such as non-linearity, differential distribution table (DDT), strict avalanche criterion (SAC), bit independence criterion (BIC), and entropy analysis. Experimental results typically show high non-linearity values, low differential probability, strong avalanche effect, and near-ideal entropy, confirming its robustness. Compared to conventional S-boxes, the logistic chaotic map-based S-box provides improved security, dynamic behavior, and key dependency, making it highly suitable for modern lightweight and high-security cryptographic applications.
