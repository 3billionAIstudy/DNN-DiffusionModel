## Introduction

Diffusion model은 Sohl-Dickstein, J.의 2015년 논문[^1]과 Y. Song의 2019년 논문[^2]을 시작으로 많은 영역에서 SOTA를 갱신하고 있는 근래에 가장 핫한 Generative model입니다.
두 논문에서는 Image를 생성했지만, 이후 더 다양한 데이터 형태의 Diffusion으로 확대해 나가면서 3D image[^3], audio[^4], video[^5], Protein[^6][^7], Molecule[^8] 나아가 (무한 차원인) 함수[^9]을 생성하는데 쓰이고 있습니다. 
이번 글에서는 이렇게 핫한 Diffusion model의 일반적인 구조를 설명해보고자 합니다. 

[^1]: Sohl-Dickstein, J. et al. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" *PMLR* (2015)

[^2]: Yang Song, and Stefano Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution", *NeurIPS* (2019)

[^3]: Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. "Dreamfusion: Text-to-3d using 2d diffusion". *arXiv* (2022)

[^4]: Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. "Diffwave: A versatile diffusion model for audio synthesis". *arXiv* (2020)

[^5]: Vikram Voleti, Alexia Jolicoeur-Martineau, and Christopher Pal. "Mcvd: Masked conditional video diffusion for prediction, generation, and interpolation". *NeurIPS* (2022)

[^6]: Kevin E Wu, Kevin K Yang, Rianne van den Berg, James Y Zou, Alex X Lu, and Ava P Amini. "Protein structure generation via folding diffusion". *arXiv* (2022)

[^7]: Joseph L. Watson et al. "Broadly applicable and accurate protein design by integrating structure prediction networks and diffusion generative models". *bioRXiv* (2022)

[^8]: Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. "Geodiff: A geometric diffusion model for molecular conformation generation". *arXiv* (2022)

[^9]: Jae Hyun Lim et al. "Score-based diffusion models in function space". *arXiv* (2023)


## Diffusion model의 기본 구조
Diffusion model은 간단히 말하면 '`Data distribution`의 `reverse diffusion`을 연산해 random initialization에서 synthetic data를 `만들어내는 모델`(Generative model)'이라 할 수 있습니다. 각각의 키워드에 대해서 자세히 살펴봅시다. 

### Data distribution and Generative model
우리가 A를 만들어내고 싶다면 먼저 명확히 해야하는 것이 크게 두 가지 있습니다.
첫번째는 'A는 어떤 공간의 원소인지'에 대한 답을 알아야합니다.
예를 들어, 이미지는 어떤 공간에 속할까요? 
디지털 관점에서 이미지란 각 pixel의 RGB 정보를 담은 벡터입니다.
따라서 $w \times h$ 크기 이미지는 $(0, 1, \cdots, 255 )^{3wh}$의 원소일 것입니다.
RGB 정보를 255로 나누어 $[0, 1]$ 사이 실수로 맵핑한다면 이미지는 $[0, 1]^{3wh}$의 원소로도 볼 수 있습니다.
Audio는 time step $t = 1, \cdots, T$에서의 소리신호 $a_t$의 형태로 이뤄져있으니 $\mathbb{R}_+^{T}$의 원소이고,
분자는 $N$개 종류의 원자들이 node가 되고, 결합을 edge로 하는 임의의 사이즈를 가진 네트워크로 볼 수 있으므로 원자 개수 $n$개의 분자는 $( 1, \cdots, N )^n ( 0, 1, 2, 3 )^{n(n - 1)/2}$의 원소가 됩니다.

그렇다면 공간 $[0, 1]^{3wh}$의 모든 벡터는 이미지라고 할 수 있을까요?
$\mathbb{R}_+^{T}$의 모든 원소는 소리일까요?
$( 1, \cdots, N )^n ( 0, 1, 2, 3 )^{n(n - 1)/2}$의 원소는 항상 어떤 분자에 대응된다고 할 수 있을까요?
그렇지 않을겁니다.
단적으로 아래의 왼쪽 그림을 두고 우리는 이미지라고 하지 않습니다.

Random noise              |  Real Image
:-------------------------:|:-------------------------:
![noise_image](https://key262yek.github.io/assets/images/noise_image.png)  |  ![real_image](https://key262yek.github.io/assets/images/real_image_cifar-10.jpg)

이것은 두 번째 질문 '공간 속 어떤 원소를 A라 부를 것인가'와 관련있습니다.
만약 A가 만족해야 하는 `조건`이 있고 조건을 만족하는 모든 데이터를 A라 부를 수 있다면, 
우리는 조건을 만족하는 임의의 데이터를 만들어냄으로써 A를 생성할 수 있을 것입니다.

하지만 만약 조건을 명확히 하기 어려운 상황이라면 어떻게 해야할까요?
'그림' 혹은 '사진'이 만족해야하는 조건은 무엇일까요?
당장에 제시하기 어렵다면, 적절한 조건을 찾아내는 방법에는 어떤 것이 있을까요?

기계학습은 조건을 학습 데이터로부터 기계 스스로 찾도록 유도하는데, 
우리는 이를 두고 '기계가 `데이터 분포`를 학습한다'고 이야기합니다.
쉽게 풀어 설명하자면 `A로 판단된 데이터`가 전체 공간 중 어디에 위치하는지를 학습하고, 나중에 데이터가 A일 확률이 높은 위치에 있다면 A로 보는 방식을 쓰겠단거죠.

여기에는 한 가지 중요한 가정이 들어가는데 `A의 학습 데이터와 비슷하면 A다`는 가정입니다.
예를 들어, 우리가 이미지라고 인식하는 데이터에서 픽셀 몇 개의 값을 조금 바꾸거나, 전체에 매우 작은 noise를 끼우더라도 우리는 그 결과도 이미지라고 인식합니다. 이 점에 착안해 학습 데이터가 몰려있는 위치 주변의 데이터, 즉 학습 데이터와 가까운 데이터를 만들어내도록 학습하면 우리는 꽤 그럴듯한 가짜 데이터를 만들어낼 수 있게 되는거죠.
(만약 데이터가 조금만 바껴도 판단이 쉽게 바뀌는 경우에는 데이터 군의 영역이 작아지고 경계가 많아지므로, 데이터 군 사이의 경계를 학습하는 모델의 학습 데이터가 그만큼 더 많이 필요하게 됩니다)

다만 이 방법은 학습 데이터의 분포에 크게 영향을 받기 때문에, 원하는 데이터의 모분포를 충분히 모사할 수 있도록 bias가 없는 학습 데이터를 구성하는 것이 중요합니다.

이런 관점에서 Generative model은 큰 틀에서 'random number generator'입니다.
앞서 학습한 data distribution을 따르는 임의의 숫자를 만들어내는 모델인 것이죠.

### Reverse diffusion
Diffusion model이 random data를 만들어내는 방법의 핵심은 reverse diffusion입니다. 
확산 과정을 역행한다는 것인데 이게 무슨 의미일까요?
먼저 데이터의 확산이 무엇인가를 먼저 생각해봅시다.

물리적으로 확산이란 밀도가 높은 곳에서 낮은 곳으로 일어나는 입자들의 흐름을 의미합니다.
모든 입자가 밀도에 의존하는 운동을 하는 것은 아니고, 미시적으로 모든 입자는 그저 무작위적으로 움직입니다.
즉 밀도가 높은 곳에서 낮은 곳으로 가는 입자가 있고, 반대로 움직이는 입자가 있습니다.
하지만 밀도 차이에 의해 전자가 훨씬 더 많아서 거시적으로 확산이란 형태로 나타나는거죠.

이를 데이터의 관점에 대응시키기 위해, 앞서 설명했던 '데이터의 공간'을 상상하고 그 안에 수많은 데이터들이 각자 점으로 표시되어 있다고 가정해봅시다. 
그리고 그 데이터에 매우 작은 noise를 주어 데이터 공간 속에서 미세하게 움직이도록 해봅시다.
이를 반복한다면 데이터는 데이터 공간 속에서 점차 확산되어 퍼질 것이고, 결국에는 모든 공간을 균등한 비율로 덮게 될 것입니다.

(Github markdown에서는 interactive plot을 바로 보여주지 못하는 관계로 링크로만 그림을 갈음합니다.
[Interactive plot link](https://key262yek.github.io/programming/Concept_Diffusion_model/#reverse-diffusion)


확산의 과정을 거치면서 각 데이터는 데이터 공간 속 임의의 점과 (확률적으로) 대응된다고 볼 수 있습니다.
만약 우리가 확산을 역행할 수 있다면, 앞선 결과는 데이터 공간 속 임의의 점으로부터 유효한 데이터를 만들어낼 수 있음을 의미합니다.
Diffusion model은 정확히 이런 과정을 통해 데이터를 생산합니다.
먼저 데이터 공간 속 확산을 규정하고, 
reverse diffusion을 딥러닝 모델을 통해 구현하는거죠.
Reverse diffusion의 방법은 모델마다 조금씩 다른데 다음 section에서 보다 자세히 알아보도록 합시다. 

## Denoising mechanism
Reverse diffusion, 내지는 'Denoising'은 크게 두 가지 방법이 있습니다. 
Sohl-Dickstein[^1]의 논문에서는 Denoising 자체를 DNN에 학습시켜 처리하고 있고,
Yang Song[^2]의 논문에서는 Langevin Monte Carlo method를 이용해 sampling하고 있습니다.
이번 section에서는 각각의 방법의 수학적 배경과 구현 방법에 대해서 설명합니다.

### Reverse diffusion kernel 
Sohl-Dickstein[^1]의 방법론의 핵심은 '작은 노이즈는 역행할 수 있다'는 점입니다.
아래 예시는 이미지에 눈에 보이지 않는 작은 노이즈를 추가해 GAN의 성능을 크게 망가뜨리는 예시인데요.
GAN의 예외적인 작동 오류에서 벗어나서 생각해보면, 작은 노이즈를 더하는 것만으로는 이미지를 '이미지가 아닌 것'으로 만들지도 못하고, 심지어는 노이즈를 가하기 전의 사진이 무엇인지조차도 명확하게 볼 수 있음을 확인할 수 있습니다.
즉, 이 정도 노이즈는 '역행 가능한 노이즈'라는 것이죠.
<img src="https://key262yek.github.io/assets/images/small_noise_image.png" alt="drawing"/>

아래 그림처럼 멀쩡한 이미지에 조금씩 노이즈를 추가하는 과정을 생각해봅시다.
총 $T$ step에 걸쳐 노이즈가 추가되면, 멀끔했던 사진은 (매우 작은 확률로 아직도 사진처럼 보일 수 있지만) 사진이라 보기 힘든 noisy rgb vector가 될 것입니다.
노이즈가 가득낀 사진을 한 번에 원래 이미지로 돌리는 것은 절대 불가능해보입니다.
하지만 각각의 작은 노이즈들이 모두 역행 가능하고, DNN에게 이를 가르칠 수 있다면 noisy rgb vector로부터 이미지를 복원해내는 것이 가능해집니다.

<img src="https://key262yek.github.io/assets/images/diffusion_model_forward_process.png" alt="drawing"/>
 <img src="https://key262yek.github.io/assets/images/diffusion_model_reverse_process.png" alt="drawing"/>

주요한 아이디어는 설명했으니 좀 더 자세히 들어가보자면, '역행 가능하다'는 것은 Reverse diffusion의 kernel이 explicit하게 정해진다는 것을 의미합니다.
예를 들어, Forward process(노이즈를 더하는 작업)에서 작은 정규분포 노이즈를 썼다면,
Reverse process도 적절한 평균과 분산을 가지는 정규분포 노이즈로 쓸 수 있습니다.
수식으로 적자면 아래와 같습니다.

$$
\begin{align}
  q(x_t | x_{t-1}) &\sim \mathcal{N}(x_t ; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbb{I}) \\
  p(x_{t-1} | x_t) &\sim \mathcal{N}(x_t ; f_\mu(x_t, t), f_\Sigma(x_t, t)) \\
\end{align}
$$

$t-1$번째 데이터에서 노이즈릴 더해 $x_t$로 만드는 Forward process kernel $q(x_t \| x_{t-1})$이 위와 같은 정규분포라면, Reverse process kernel $p(x_{t-1}\| x_t)$도 정규분포를 따른다는 의미이고, 이 과정에서 시점 $t$에서의 diffusion size $\beta_t$와 Reverse process의 평균과 분산을 결정하는 $f_\mu, f_\Sigma$ 함수는 모두 DNN을 이용해 결정해야하는 변수들입니다.

학습에 사용될 Loss는 log likelihood $\mathbb{E}_{q(x_0)}[-log p(x_0)]$인데 이는 바로 계산할 수 없은 영역이라 variational bound를 정의해 이용합니다.

$$
\begin{equation}
  \mathbb{E}[- \log p(x_0)] \leq 
  \mathbb{E}_q [- \log p(x_T) - \sum_{t \geq 1} \log \frac{p(x_{t-1} | x_t)}{q(x_t | x_{t-1})}]
\end{equation}
$$

### Score-based model
Yang Song의 Score-based model은 Langevin Monte Carlo method를 이용해서 denoise를 합니다.
Langevin Monte Carlo method는 전형적은 Markov Chain Monte Carlo method 중 하나로,
Langevin equation

$$
\begin{equation}
  dX(t) = - \frac{1}{\gamma} \nabla V(X) dt + \sqrt{2 D} dW 
\end{equation}
$$

를 만족하며 변하는 입자의 평형 분포 $P(x)$가 아래와 같이 주어짐을 이용하는 샘플링 방법론입니다.

$$
\begin{equation}
  P(x) \propto \exp (- \frac{V(x)}{k_B T}), \quad \gamma D = 2 k_B T
\end{equation}
$$

즉, 원하는 확률분포 $P(x)$가 있을 때, Potential 항 $V(x)$를 $-C \log P(x)$로 적고 아래와 같은 Langevin equation을 따르도록 시뮬레이션한다면 그 결과는 최종적으로 $P(x)$ 분포를 따르는 확률변수가 될 것이라는 점을 이용합니다.

$$
\begin{equation}
  d X(t) = \frac{1}{\gamma} \nabla \log P(x) dt + \sqrt{2 D} dW 
\end{equation}
$$

따라서 이 모델에서 DNN은 $\nabla \log P(x)$를 학습하도록 설계되어 있으며,
inference는 random vector의 Langevin simulation을 몇 단계 수행한 후 결과를 내뱉는 방식을 띄고 있습니다. 

이때 DNN의 Loss는 $\nabla \log p(x)$와의 $L_2$-norm의 기댓값을 최소화하도록 주어지는데,
이것 역시 바로 계산될 수 있는 형태가 아니기 때문에 equivalent한 다른 형태의 Loss를 대신 사용합니다. 

$$
\begin{equation}
  L = \mathbb{E} [ tr(\nabla s(x)) + \frac{1}{2} |s(x)|^2 ]
\end{equation}
$$

## Limitation
Diffusion model은 [Stable diffusion](https://stablediffusionweb.com/),
[Diffdock](https://github.com/gcorso/DiffDock),
[RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) 등 다양한 영역에서 성공적인 성과를 보여주고 있지만, 
분명한 단점들이 있습니다.

첫 번째로 모델 설정에 수학적인 배경지식이 매우 많이 필요합니다.
데이터 구조에 따라서 사용할 수 있는 적절한 parameter가 주어질텐데, 
이 parameter의 diffusion이 항상 정규분포 꼴일 필요가 없습니다.
예를 들어, Diffdock은 분자의 구성 원소들의 위치를 diffusion해서 단백질과의 결합위치를 찾는 
global docking 모델인데요. 
이 과정에서 구성 원소들의 위치를 크게 평행이동, 회전, torsion으로 구분하고 회전과 torsion은 평행이동과는 전혀 다른 형태의 diffusion kernel을 정의해야만 합니다. 
Function space에서의 diffusion을 생각할 때는 이보다도 더 심각해지겠죠.

두 번째 한계는 모델의 feature 수와 inference time입니다.
두 모델 모두 세부적으로 뜯어보면 각 Time step 별로 reverse diffusion kernel과 score 네트워크를 달리 쓰는데요. 
Diffusion model에서 보통 사용하는 Time step 수가 많게는 1000개까지 간다는 점을 생각하면 상당히 많은 feature를 학습해야한다는 문제가 있습니다.
또한 그렇게 큰 모델을 inference 하는 것 역시 시간이 많이 소요되는 작업이구요.

그래서 Diffusion을 좀 더 적은 time step으로 나타내고, inference를 빠르게 하는 다른 방법들 역시 연구되고 있습니다. 


## References
