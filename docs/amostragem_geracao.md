# P3 – Resultados, observações, dificuldades e limitações

Registro da P3 (amostragem, geração e áudio) para as seções de resultados, discussão e limitações do artigo.
---

## 1. Resultados e observações

### 1.1 Validação da representação (teste de ida e volta)
- Tokens do conjunto de teste decodificados para MIDI e comparados com os MIDIs originais pré-processados (altura + início da nota).
- **Resultado final: 88 de 88 clipes com 100% das notas recuperadas** (média e mínimo = 100%).
- Antes da correção dos dados (ver Dificuldades, item 2.1), o mesmo teste recuperava **só 0–0,4% das notas**.

Frase sugerida (reescrever com as palavras do grupo):
> O teste de ida e volta nos 88 clipes do conjunto de teste recuperou todas as notas com a mesma altura e o mesmo início, confirmando que a representação não perde informação de altura e posição.

### 1.2 Comprimento das sequências por quadrante (conjunto de teste)

| Quadrante | Clipes | Média | Mediana | Desvio | Mín–Máx |
|---|---|---|---|---|---|
| Q1 | 25 | 1.307 | 1.211 | 398 | 747–2.068 |
| Q2 | 19 | 1.596 | 1.630 | 863 | 246–3.851 |
| Q3 | 21 | 842 | 908 | 347 | 272–1.594 |
| Q4 | 23 | 946 | 886 | 279 | 594–1.498 |

**Observação:** os quadrantes de **alta excitação (Q1, Q2) têm sequências cerca de 50% mais longas** que os de baixa excitação (Q3, Q4), ou seja, mais eventos no mesmo tempo. Isso é coerente com a associação entre excitação e densidade de notas relatada no EMOPIA (Hung et al., 2021), e é um primeiro indício de que a densidade é uma característica forte para distinguir a excitação. Q2 também tem a maior variabilidade (desvio de 863).

Esses comprimentos foram usados para sortear o tamanho de cada música gerada, para que a distribuição de tamanhos das músicas geradas fosse igual à das humanas.

### 1.3 Geração com o modelo final
- **Configuração de amostragem:** temperatura 1,0, top-p 0,95, sem top-k; semente 42; lote de 25 músicas.
- **Músicas geradas:** 100 (25 por quadrante), com comprimentos sorteados a partir da distribuição do conjunto de teste de cada quadrante.
- **Vazias:** 0 / 100.
- **Terminaram com EOS:** 0 / 100. **Todas pararam no comprimento máximo sorteado.**
- **Tempo de geração:** cerca de 64 s para as 100 músicas, em CPU (notebook Acer Aspire A514-54, sem GPU).
- **Proporção de notas por token:** 0,21–0,22 em todos os quadrantes (tabela abaixo), igual à das músicas humanas (ex.: clipe de teste `Q1_4dXC1cC7crw_0`, com 251 notas em 1.190 tokens). Indica que o modelo gera sequências REMI bem formadas, que o decodificador aproveita quase por inteiro.

**Observação sobre o EOS:** o modelo nunca encerrou a peça sozinho. Provável explicação: o treino usa janelas de cerca de 460 tokens, e só a última janela de cada clipe contém o EOS, então o token de fim aparece raramente no treino. Na prática, o comprimento das músicas geradas é definido pelo sorteio a partir do teste, e não pelo modelo.

Estatísticas por quadrante (fonte: `amostras_geradas/lstm/metadados.csv`):

| Quadrante | n | Notas (média) | Tokens (média) | Tokens no teste (média) | Notas por token | < 20 notas |
|---|---|---|---|---|---|---|
| Q1 | 25 | 285 | 1.289 | 1.307 | 0,221 | 0 |
| Q2 | 25 | 305 | 1.367 | 1.596 | 0,223 | 0 |
| Q3 | 25 | 169 | 792 | 842 | 0,213 | 0 |
| Q4 | 25 | 196 | 922 | 946 | 0,213 | 0 |
| **Total** | **100** | **239** | **1.092** | 1.164 | 0,219 | **0** |

**Observações:**
- **Comprimentos próximos aos do teste**, como pretendido pelo sorteio. Q2 ficou abaixo (1.367 contra 1.596) por acaso do sorteio: o teste de Q2 tem só 19 clipes e alta variabilidade (246 a 3.851 tokens).
- **Q1 e Q2 têm bem mais notas que Q3 e Q4**, mas **isso não prova que o condicionamento funciona**: a diferença vem quase toda do comprimento imposto pelo sorteio, que já é maior em Q1/Q2. Prova disso é que a proporção de notas por token é praticamente igual em todos os quadrantes (0,21–0,22).
- Há uma diferença pequena na proporção de notas por token: um pouco maior nos quadrantes de alta excitação (0,221–0,223) que nos de baixa (0,213). Isso pode indicar mais notas simultâneas (acordes) ou menos tokens de posição/tempo em Q1/Q2, mas é pequena demais para conclusões.
- **Para avaliar o efeito da emoção na densidade, é preciso medir notas por segundo ou por compasso** (métricas do P4), e não o número total de notas.

Estudo de amostragem: [não realizado por falta de tempo / realizado: configurações e resultado].

Se houver estudo de amostragem: como as métricas e a diferença entre quadrantes mudam com a temperatura. Hipótese: temperaturas altas aumentam a diversidade, mas diminuem a diferença entre emoções.

### 1.4 Análise qualitativa (audição)
Audição informal das músicas geradas, por um integrante do grupo (sem teste às cegas). Impressão geral: resultado **razoável**, soa como piano com alguma coerência.

| Quadrante | Emoção esperada | Impressão |
|---|---|---|
| Q1 | alegre / empolgado | **Soa alegre**, como esperado |
| Q2 | tenso / agitado | Agitado, mas **parecido com Q1**: a tensão não aparece |
| Q3 | triste / melancólico | **Soa triste**, mas ainda um pouco agitado |
| Q4 | calmo / sereno | Menos agitado que os outros, mas **não chega a soar calmo** |

**Leitura em termos de valência e excitação:**
- **Valência:** distinguida nos quadrantes de **baixa excitação** (Q3 triste × Q4 não triste), mas **não nos de alta excitação** (Q1 e Q2 soam parecidos, os dois "alegres e agitados"). O modelo parece não ter aprendido o que torna uma música agitada *tensa* em vez de *alegre*: harmonia dissonante, modo menor, registro.
- **Excitação:** a direção está certa (Q4 é o menos agitado, Q1/Q2 os mais), mas a faixa parece **comprimida para cima**: Q3 e Q4 ainda soam agitados demais. Ou seja, o modelo não chega à calma das músicas humanas de baixa excitação.
- **Em relação à hipótese** de que a excitação seria mais fácil de reproduzir que a valência: a impressão é **parcialmente contrária**. A valência foi percebida no par Q3/Q4, e a excitação ficou comprimida. Isso deve ser verificado com as métricas por quadrante do P4 (densidade de notas, velocity, modo maior/menor), porque a audição informal é limitada.

**Cuidados ao citar:** um único ouvinte, poucas músicas, sem ocultar o quadrante (o ouvinte sabia qual emoção esperar, o que pode influenciar a percepção). Tratar como observação qualitativa, não como resultado.

---

## 2. Dificuldades

### 2.1 Desalinhamento entre os tokens e o tokenizer (a principal)
- **Sintoma:** o teste de ida e volta recuperava só 0–0,4% das notas, com bem menos notas que o original (um clipe de 901 notas voltou com 5).
- **Como foi percebido:** as alturas decodificadas estavam **sempre 4 semitons abaixo** das originais (32 → 28, 40 → 36, 37 → 33). O deslocamento constante de 4 coincidia com o número de tokens de emoção adicionados ao vocabulário.
- **Causa:** os arquivos de tokens tinham sido gerados com um vocabulário de 413 tokens, sem emoções, e as emoções foram acrescentadas no fim (IDs 413–416). Já o tokenizer salvo tinha as emoções nos IDs 4–7, deslocando todos os outros tokens. Lida com o tokenizer salvo, uma sequência começava como `Program_125, Emotion_Q1` em vez de `Emotion_Q1, Bar`.
- **Impacto potencial:** se o modelo fosse treinado e decodificado com vocabulários diferentes, as músicas geradas sairiam erradas, e o problema poderia passar despercebido.
- **Solução:** dados regenerados pelo P1 (`processed_tokens_convertidos/`), teste refeito com 100%, e confirmação de que o modelo final foi treinado com a pasta corrigida (registrado no checkpoint).
- **Lição:** o teste de ida e volta deveria fazer parte do pipeline desde o início, rodando sempre que os tokens forem regenerados.

### 2.2 Coexistência de duas pastas de tokens
- Durante a correção, `processed_tokens/` (antiga) e `processed_tokens_convertidos/` (corrigida) coexistiram. O checkpoint registra o tokenizer na pasta antiga, e a primeira execução do script de geração leu a pasta errada (erro "primeiro token inesperado: 413").
- Resolvido apontando explicitamente para a pasta corrigida (`--pasta-tokens`). O ideal seria manter uma pasta só.

### 2.3 Erros encontrados na implementação da amostragem
- `top_k = 0` zerava todos os logits e gerava NaN no softmax → passou a significar "filtro desligado".
- O mascaramento do top-p usava `scatter` com índices repetidos, cujo resultado não é garantido (sobretudo na GPU) → trocado por `scatter` com a permutação ordenada.
- O token MASK não estava na lista de proibidos de início → incluído, para não ser sorteado durante a geração.
- Todos os casos foram cobertos por testes no próprio `amostragem.py`.

### 2.4 Ambiente e hardware
- O checkpoint foi salvo numa GPU (Tesla T4), e a máquina da P3 não tem GPU. Carregar exige `map_location='cpu'`, e a geração na CPU é mais lenta.
- Detalhes da API do miditok 3.1 (formato de entrada do `decode` depende de `one_token_stream`) exigiram verificação antes da decodificação.

### 2.5 Dependências e prazo
- A geração dependia do modelo treinado (P2), que por sua vez dependia da correção dos dados (P1). Com isso, a geração final só pôde ser feita perto do prazo, e o estudo de amostragem foi [reduzido / não realizado].

---

## 3. Limitações

### 3.1 Comprimento gerado além da janela de treino
- As janelas de treino têm **cerca de 460 tokens** em média (7.309.783 tokens / 15.896 janelas), mas as músicas geradas têm de 250 a 3.850 tokens (média por quadrante entre 840 e 1.600), para igualar a distribuição do teste. Ou seja, 2 a 3 vezes o contexto visto no treino.
- A LSTM mantém o estado entre os passos, mas a coerência e a emoção podem se perder no fim das músicas longas, sobretudo em Q1 e Q2.

### 3.2 Condicionamento só pelo primeiro token
- O modelo final usa `condicionamento = "token"`: a emoção aparece uma única vez, no início. Em sequências longas, a influência desse token pode se diluir com o tempo (esquecimento da LSTM).
- Possível análise: comparar as métricas no início e no fim das músicas geradas (P4).

### 3.3 Escolha da amostragem
- [Se não houver estudo] Configuração fixa (T = 1,0, top-p = 0,95), sem busca sistemática. Os resultados podem mudar com outras configurações.
- [Se houver] Estudo pequeno (3 configurações, 20 músicas cada), escolhido na validação.

### 3.4 Perdas da representação (não detectadas pelo teste de ida e volta)
- O teste verificou altura e início. A **velocity** é agrupada em 32 faixas, e o **andamento** em 32 valores entre 40 e 250 BPM.
- A grade de tempo é de semicolcheia (4 posições por tempo): **tercinas e microvariações expressivas de tempo são arredondadas**. Isso pode tirar do piano a expressividade associada à emoção (rubato, por exemplo).

### 3.5 Áudio
- Renderização com soundfont General MIDI (FluidR3_GM), menos realista que um piano gravado. A percepção da emoção na audição pode ser afetada pelo timbre. Os áudios servem só para audição, não para a avaliação objetiva.

### 3.6 O modelo não aprende a terminar a música
- Nenhuma das 100 músicas terminou com EOS: o fim é sempre um corte no comprimento sorteado. As peças geradas podem terminar no meio de uma frase musical, sem cadência final.

### 3.7 Reprodutibilidade
- Com a mesma semente, a geração é reproduzível, mas o resultado também depende do tamanho do lote (`--lote`), que precisa ser mantido igual.

---