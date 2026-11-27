# pMF Any-to-Any com RGB e bounding boxes

Abre **notebooks/pmf_rgb_boxes_colab.ipynb** numa T4. Esta variante usa
**configs/pmf_rgb_boxes.json** e um diretório novo de resultados. O RF100
Corrosion Bi3Q3 é carregado a 512×512 com os splits originais 840/304/105.
As únicas modalidades ativas são RGB e caixas. O teste fica fora da otimização
e da validação por época. A variante anterior continua disponível.

## Auditoria e limite de paridade

- O loader reutiliza o pMF-L/32-512 da revisão fixada pelo baseline e
  rejeita pesos em falta, incompatíveis ou inesperados neste protocolo.
  As projeções LoRA e o residual RGB partem de contribuição zero.
- Os tokens RGB mantêm os embeddings posicionais e o RoPE do pMF.
  Os tokens de caixas usam posições próprias antes da grelha RGB.
  A atenção e os blocos Transformer são os do checkpoint.
- O teste inicial compara os pixels pré-clipping do modo exclusivamente RGB
  com um passo Euler do Transformer original com o mesmo ruído, classe
  ImageNet, guidance 7.5 e intervalo 0.2–0.6. A condição de classe original
  serve para testar paridade. O treino RF100 usa a condição ImageNet nula.
- Usa-se z(t) = (1-t)x + tε com um intervalo t,r partilhado. O JVP move os
  estados-alvo em ε-x e o tempo t em 1. A condição e r têm tangente zero.
  A loss usa u + (t-r) stopgrad(du/dt). O teste compara o JVP RGB com
  diferenças centrais em FP32. Permanece a adaptação ao checkpoint de
  inferência, que não fornece a cabeça v do treino completo do pMF.
- A configuração original L/32 usa ruído RGB com escala 4, tempos
  logit-normal com desvio 0,8 e normalização adaptativa com expoente 1 e
  epsilon 0,01. O protocolo anterior usava expoente 0,5, que permite que
  o número mostrado continue a crescer com o resíduo RGB. O protocolo novo
  usa a normalização original e regista o MSE por coordenada antes da
  ponderação para detetar divergência que a loss ponderada pode ocultar.
- As caixas usam coordenadas xyxy em [0,1]. A geometria é MSE por
  coordenada de caixa presente. A cross-entropy existente classifica os
  slots, incluindo ausência. O relatório expõe MSE, CE e a soma de ambos.
  A loss global faz a média dos alvos presentes na tarefa. O protocolo
  não usa máscaras de edição, perdas perceptuais ou outros termos novos.
- O overfit usa quatro imagens de treino e seeds de ruído fixas. O gate
  exige losses e gradientes finitos, gradiente não nulo e descida da loss
  nas quatro tarefas. Regista antes/depois e pico de VRAM. Os pesos desse
  teste são descartados. O treino começa num novo processo, no checkpoint
  original. A paridade é uma propriedade da inicialização. A qualidade
  RGB depois das atualizações é medida por época.

## Validação e retoma

Na época 0, antes de atualizar pesos, calcula-se loss de validação, FID, KID,
mAP, latência e amostras. Não há loss de treino nem gradientes nessa época.
As frequências de validação são balanceadas entre as quatro tarefas.
A partir da época 1 o treino amostra as tarefas uniformemente e reporta
as frequências observadas, a norma pré-clipping dos gradientes e o LR.

FID e KID usam as 304 imagens de validação e as gerações correspondentes.
São reportados por tarefa RGB. O mAP usa RGB real → caixas previstas, contra
as anotações. As gerações livres não têm uma caixa real pareada. A latência
é medida à volta do sampler, com sincronização CUDA, excluindo a primeira
imagem de cada tarefa. As imagens qualitativas usam duas referências fixas,
seeds fixas, verde para caixas anotadas e rosa para caixas previstas.

O notebook atualiza curvas depois de cada época. **RUN/epochs** guarda os
JSON, **RUN/examples** as amostras e curvas, e **RUN/checkpoints** os pesos
treináveis, o otimizador e os RNGs desde a época 0. Define RESUME para o
checkpoint aplicável e conserva o mesmo config e dataset. Os ficheiros
em /content são temporários e podem ser exportados sem o Drive.

Referências: [configuração L/32](https://github.com/Lyy-iiis/pMF/blob/main/configs/pMF_L_32_config.yml), [defaults](https://github.com/Lyy-iiis/pMF/blob/main/configs/default.py) e [objetivo MeanFlow](https://github.com/Lyy-iiis/pMF/blob/main/pmf.py).
