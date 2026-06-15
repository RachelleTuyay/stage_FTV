---
language:
- en
- multilingual
- ar
- bg
- ca
- cs
- da
- de
- el
- es
- et
- fa
- fi
- fr
- gl
- gu
- he
- hi
- hu
- hy
- id
- it
- ja
- ka
- ko
- ku
- lt
- lv
- mk
- mn
- mr
- ms
- my
- nb
- nl
- pl
- pt
- ro
- ru
- sk
- sl
- sq
- sr
- sv
- th
- tr
- uk
- ur
- vi
- zh
- hr
license: apache-2.0
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:62698210
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: A man is jumping unto his filthy bed.
  sentences:
  - A man is ouside near the beach.
  - The bed is dirty.
  - The man is on the moon.
- source_sentence: Ship Simulator (video game)
  sentences:
  - ಯಂತ್ರ ಕಲಿಕೆ
  - Ship Simulator
  - جان بابتيست لويس بيير
- source_sentence: And so was the title of his book on the Israeli massacre of Gaza
    in 2008-2009.
  sentences:
  - Antony Lowenstein ist ein bekannter Blogger über den Nahen Osten.
  - Y ese fue el título de su libro sobre la masacre israelí de Gaza entre 2008 y
    2009.
  - 'C''était au temps où vous ne pouviez pas avoir un film de Nollywood qui n''incluait
    pas un ou une combinaison des aspects suivants: fraude, gris-gris/sorcellerie,
    vol à main armée, inceste, adultère, cannibalisme et, naturellement notre sujet
    favori, la corruption.'
- source_sentence: In fact, it contributes more than 12 percent to Thailand’s GDP.
  sentences:
  - Einige Provider folgten der Anordnung, aber „Fitna“ konnte noch über andere Anbieter
    angesehen werden.
  - En fait, il représente plus de 12% du produit national brut thaïlandais.
  - '"Aber von heute an...heute ist der Anfang eines neuen Lebens für mich."'
- source_sentence: It is known for its dry red chili powder .
  sentences:
  - These monsters will move in large groups .
  - It is popular for dry red chili powder .
  - In a statistical overview derived from writings by and about William George Aston
    , OCLC/WorldCat includes roughly 90 + works in 200 + publications in 4 languages
    and 3,000 + library holdings .
datasets:
- sentence-transformers/parallel-sentences-wikititles
- sentence-transformers/parallel-sentences-tatoeba
- sentence-transformers/parallel-sentences-talks
- sentence-transformers/parallel-sentences-europarl
- sentence-transformers/parallel-sentences-global-voices
- sentence-transformers/parallel-sentences-muse
- sentence-transformers/parallel-sentences-wikimatrix
- sentence-transformers/parallel-sentences-opensubtitles
- sentence-transformers/stackexchange-duplicates
- sentence-transformers/quora-duplicates
- sentence-transformers/wikianswers-duplicates
- sentence-transformers/all-nli
- sentence-transformers/simple-wiki
- sentence-transformers/altlex
- sentence-transformers/flickr30k-captions
- sentence-transformers/coco-captions
- sentence-transformers/nli-for-simcse
- jinaai/negation-dataset
pipeline_tag: sentence-similarity
library_name: sentence-transformers
co2_eq_emissions:
  emissions: 196.7083299812303
  energy_consumed: 0.5060646201491896
  source: codecarbon
  training_type: fine-tuning
  on_cloud: false
  cpu_model: 13th Gen Intel(R) Core(TM) i7-13700K
  ram_total_size: 31.777088165283203
  hours_used: 3.163
  hardware_used: 1 x NVIDIA GeForce RTX 3090
---

# Static Embeddings with BERT Multilingual uncased tokenizer finetuned on various datasets

This is a [sentence-transformers](https://www.SBERT.net) model trained on the [wikititles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikititles), [tatoeba](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba), [talks](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks), [europarl](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-europarl), [global_voices](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices), [muse](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-muse), [wikimatrix](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix), [opensubtitles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles), [stackexchange](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates), [quora](https://huggingface.co/datasets/sentence-transformers/quora-duplicates), [wikianswers_duplicates](https://huggingface.co/datasets/sentence-transformers/wikianswers-duplicates), [all_nli](https://huggingface.co/datasets/sentence-transformers/all-nli), [simple_wiki](https://huggingface.co/datasets/sentence-transformers/simple-wiki), [altlex](https://huggingface.co/datasets/sentence-transformers/altlex), [flickr30k_captions](https://huggingface.co/datasets/sentence-transformers/flickr30k-captions), [coco_captions](https://huggingface.co/datasets/sentence-transformers/coco-captions), [nli_for_simcse](https://huggingface.co/datasets/sentence-transformers/nli-for-simcse) and [negation](https://huggingface.co/datasets/jinaai/negation-dataset) datasets. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, paraphrase mining, text classification, clustering, and more.

Read our [Static Embeddings blogpost](https://huggingface.co/blog/static-embeddings) to learn more about this model and how it was trained.

* **0 Active Parameters:** This model does not use any active parameters, instead consisting exclusively of averaging pre-computed token embeddings.
* **100x to 400x faster:** On CPU, this model is 100x to 400x faster than common options like [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small). On GPU, it's 10x to 25x faster.
* **Matryoshka:** This model was trained with a [Matryoshka loss](https://huggingface.co/blog/matryoshka), allowing you to truncate the embeddings for faster retrieval at minimal performance costs.
* **Evaluations:** See [Evaluations](#evaluation) for details on performance on NanoBEIR, embedding speed, and Matryoshka dimensionality truncation.
* **Training Script:** See [train.py](train.py) for the training script used to train this model from scratch.

See [`static-retrieval-mrl-en-v1`](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) for an English static embedding model that has been finetuned specifically for retrieval tasks.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** inf tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Datasets:**
    - [wikititles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikititles)
    - [tatoeba](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba)
    - [talks](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks)
    - [europarl](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-europarl)
    - [global_voices](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices)
    - [muse](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-muse)
    - [wikimatrix](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix)
    - [opensubtitles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles)
    - [stackexchange](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates)
    - [quora](https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
    - [wikianswers_duplicates](https://huggingface.co/datasets/sentence-transformers/wikianswers-duplicates)
    - [all_nli](https://huggingface.co/datasets/sentence-transformers/all-nli)
    - [simple_wiki](https://huggingface.co/datasets/sentence-transformers/simple-wiki)
    - [altlex](https://huggingface.co/datasets/sentence-transformers/altlex)
    - [flickr30k_captions](https://huggingface.co/datasets/sentence-transformers/flickr30k-captions)
    - [coco_captions](https://huggingface.co/datasets/sentence-transformers/coco-captions)
    - [nli_for_simcse](https://huggingface.co/datasets/sentence-transformers/nli-for-simcse)
    - [negation](https://huggingface.co/datasets/jinaai/negation-dataset)
- **Languages:** en, multilingual, ar, bg, ca, cs, da, de, el, es, et, fa, fi, fr, gl, gu, he, hi, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh, hr
- **License:** apache-2.0

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): StaticEmbedding(
    (embedding): EmbeddingBag(105879, 1024, mode='mean')
  )
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("tomaarsen/static-similarity-mrl-multilingual-v1")
# Run inference
sentences = [
    'It is known for its dry red chili powder .',
    'It is popular for dry red chili powder .',
    'These monsters will move in large groups .',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

This model was trained with Matryoshka loss, allowing this model to be used with lower dimensionalities with minimal performance loss.
Notably, a lower dimensionality allows for much faster downstream tasks, such as clustering or classification. You can specify a lower dimensionality with the `truncate_dim` argument when initializing the Sentence Transformer model:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("tomaarsen/static-similarity-mrl-multilingual-v1", truncate_dim=256)
embeddings = model.encode([
    "I used to hate him.",
    "Раньше я ненавидел его."
])
print(embeddings.shape)
# => (2, 256)
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

We've evaluated the model on 5 languages which have a lot of benchmarks across various tasks on [MTEB](https://huggingface.co/spaces/mteb/leaderboard).

We want to reiterate that this model is not intended for retrieval use cases. Instead, we evaluate on Semantic Textual Similarity (STS), Classification, and Pair Classification. We compare against the excellent and small [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) model.

![](img/similarity_mteb_eval.png)

Across all measured languages, [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) reaches an average **92.3%** for STS, **95.52%** for Pair Classification, and **86.52%** for Classification relative to [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small).

![](img/similarity_speed.png)

To make up for this performance reduction, [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) is approximately ~125x faster on CPU and ~10x faster on GPU devices than [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small). Due to the super-linear nature of attention models, versus the linear nature of static embedding models, the speedup will only grow larger as the number of tokens to encode increases.

#### Matryoshka Evaluation

Lastly, we experimented with the impacts on English STS on MTEB performance when we did Matryoshka-style dimensionality reduction by truncating the output embeddings to a lower dimensionality. 

![English STS MTEB performance vs Matryoshka dimensionality reduction](img/similarity_matryoshka.png)

As you can see, you can easily reduce the dimensionality by 2x or 4x with minor (0.15% or 0.56%) performance hits. If the speed of your downstream task or your storage costs are a bottleneck, this should allow you to alleviate some of those concerns.

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Datasets

<details><summary>wikititles</summary>

* Dataset: [wikititles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikititles) at [d92a4d2](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikititles/tree/d92a4d28a082c3c93563feb92a77de6074bdeb52)
* Size: 14,700,458 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                       | non_english                                                                                    |
  |:--------|:----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                        | string                                                                                         |
  | details | <ul><li>min: 4 characters</li><li>mean: 18.33 characters</li><li>max: 84 characters</li></ul> | <ul><li>min: 4 characters</li><li>mean: 17.19 characters</li><li>max: 109 characters</li></ul> |
* Samples:
  | english                 | non_english                |
  |:------------------------|:---------------------------|
  | <code>Le Vintrou</code> | <code>Ле-Вентру</code>     |
  | <code>Greening</code>   | <code>Begrünung</code>     |
  | <code>Warrap</code>     | <code>واراب (توضيح)</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>tatoeba</summary>

* Dataset: [tatoeba](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba) at [cec1343](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba/tree/cec1343ab5a7a8befe99af4a2d0ca847b6c84743)
* Size: 4,138,956 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                        | non_english                                                                                    |
  |:--------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 5 characters</li><li>mean: 31.59 characters</li><li>max: 196 characters</li></ul> | <ul><li>min: 6 characters</li><li>mean: 30.95 characters</li><li>max: 161 characters</li></ul> |
* Samples:
  | english                                                | non_english                          |
  |:-------------------------------------------------------|:-------------------------------------|
  | <code>I used to hate him.</code>                       | <code>Раньше я ненавидел его.</code> |
  | <code>It is nothing less than an insult to her.</code> | <code>それはまさに彼女に対する侮辱だ。</code>        |
  | <code>I've apologized, so lay off, OK?</code>          | <code>謝ったんだから、さっきのはチャラにしてよ。</code>   |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>talks</summary>

* Dataset: [talks](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks) at [0c70bc6](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks/tree/0c70bc6714efb1df12f8a16b9056e4653563d128)
* Size: 9,750,031 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                        | non_english                                                                                    |
  |:--------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 5 characters</li><li>mean: 94.41 characters</li><li>max: 493 characters</li></ul> | <ul><li>min: 4 characters</li><li>mean: 82.49 characters</li><li>max: 452 characters</li></ul> |
* Samples:
  | english                                                                                               | non_english                                                    |
  |:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | <code>(Laughter) EC: But beatbox started here in New York.</code>                                     | <code>(Skratt) EC: Fast beatbox började här i New York.</code> |
  | <code>I did not have enough money to buy food, and so to forget my hunger, I started singing."</code> | <code>食べ物を買うお金もなかった だから 空腹を忘れるために 歌を歌い始めたの」</code>             |
  | <code>That is another 25 million barrels a day.</code>                                                | <code>那时还要增加两千五百万桶的原油。</code>                                  |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>europarl</summary>

* Dataset: [europarl](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-europarl) at [11007ec](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-europarl/tree/11007ecf9c790178a49a4cbd5cfea451a170f2dc)
* Size: 4,990,000 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                         | non_english                                                                                     |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 0 characters</li><li>mean: 147.77 characters</li><li>max: 668 characters</li></ul> | <ul><li>min: 0 characters</li><li>mean: 153.13 characters</li><li>max: 971 characters</li></ul> |
* Samples:
  | english                                                                                                                                                                                                                                                                                                                                           | non_english                                                                                                                                                                                                                                                                                                                                              |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>(SK) I would like to stress three key points in relation to this issue.</code>                                                                                                                                                                                                                                                              | <code>(SK) Chtěla bych zdůraznit tři klíčové body, které jsou s tímto tématem spojeny.</code>                                                                                                                                                                                                                                                            |
  | <code>Women have a higher recorded rate of unemployment, especially long term unemployment.</code>                                                                                                                                                                                                                                                | <code>Blandt kvinder registreres større arbejdsløshed, især blandt langtidsarbejdsløse.</code>                                                                                                                                                                                                                                                           |
  | <code>You will recall that we have occasionally had disagreements over how to interpret Rule 166 of our Rules of Procedure and that certain Members thought that the Presidency was not applying it properly, since it was not giving the floor for points of order that did not refer to the issue that was being debated at that moment.</code> | <code>De husker nok, at vi til tider har været uenige om fortolkningen af artikel 166 i vores forretningsorden, og at nogle af medlemmerne mente, at formanden ikke anvendte den korrekt, eftersom han ikke gav ordet til indlæg til forretningsordenen, når det ikke drejede sig om det spørgsmål, der blev drøftet på det pågældende tidspunkt.</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>global_voices</summary>

* Dataset: [global_voices](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices) at [4cc20ad](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices/tree/4cc20add371f246bb1559b543f8b0dea178a1803)
* Size: 1,099,099 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                         | non_english                                                                                     |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 5 characters</li><li>mean: 115.13 characters</li><li>max: 740 characters</li></ul> | <ul><li>min: 3 characters</li><li>mean: 119.89 characters</li><li>max: 801 characters</li></ul> |
* Samples:
  | english                                                                                   | non_english                                                                                                 |
  |:------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|
  | <code>Generation 9/11: Cristina Balli (USA) from British Council USA on Vimeo.</code>     | <code>Генерација 9/11: Кристина Бали (САД) од Британскиот совет САД на Вимео.</code>                        |
  | <code>Jamaica: Mapping the state of emergency · Global Voices</code>                      | <code>Jamaica: Mapeando el estado de emergencia</code>                                                      |
  | <code>It takes more than courage or bravery to do such a... http://fb.me/12T47y0Ml</code> | <code>Θέλει κάτι παραπάνω από κουράγιο ή ανδρεία για να κάνεις κάτι τέτοιο... http://fb.me/12T47y0Ml</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>muse</summary>

* Dataset: [muse](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-muse) at [238c077](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-muse/tree/238c077ac66070748aaf2ab1e45185b0145b7291)
* Size: 1,368,274 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                      | non_english                                                                                  |
  |:--------|:---------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
  | type    | string                                                                                       | string                                                                                       |
  | details | <ul><li>min: 3 characters</li><li>mean: 7.38 characters</li><li>max: 16 characters</li></ul> | <ul><li>min: 1 characters</li><li>mean: 7.33 characters</li><li>max: 18 characters</li></ul> |
* Samples:
  | english              | non_english         |
  |:---------------------|:--------------------|
  | <code>metro</code>   | <code>metrou</code> |
  | <code>suggest</code> | <code>제안</code>     |
  | <code>nnw</code>     | <code>nno</code>    |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>wikimatrix</summary>

* Dataset: [wikimatrix](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix) at [74a4cb1](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix/tree/74a4cb15422cdd0c3aacc93593b6cb96a9b9b3a9)
* Size: 9,688,498 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                          | non_english                                                                                      |
  |:--------|:-------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                           |
  | details | <ul><li>min: 16 characters</li><li>mean: 124.31 characters</li><li>max: 418 characters</li></ul> | <ul><li>min: 11 characters</li><li>mean: 129.99 characters</li><li>max: 485 characters</li></ul> |
* Samples:
  | english                                                                                                                                    | non_english                                                                                                                           |
  |:-------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|
  | <code>3) A set of wikis to support collaboration activities and disseminate information about good practices.</code>                       | <code>3) Un conjunt de wikis per donar suport a les activitats de col·laboració i difusió d'informació sobre bones pràctiques.</code> |
  | <code>Daily cruiseferry services operate to Copenhagen and Frederikshavn in Denmark, and to Kiel in Germany.</code>                        | <code>Dịch vụ phà du lịch hàng ngày vận hành tới Copenhagen và Frederikshavn tại Đan Mạch, và tới Kiel tại Đức.</code>                |
  | <code>In late April 1943, Philipp was ordered to report to Hitler's headquarters, where he stayed for most of the next four months.</code> | <code>Sent i april 1943 fick Philipp ordern att rapportera till Hitlers högkvarter, där han stannade i fyra månader.</code>           |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>opensubtitles</summary>

* Dataset: [opensubtitles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles) at [d86a387](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles/tree/d86a387587ab6f2fd9ec7453b2765cec68111c87)
* Size: 4,990,000 training samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                        | non_english                                                                                    |
  |:--------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 0 characters</li><li>mean: 34.43 characters</li><li>max: 220 characters</li></ul> | <ul><li>min: 0 characters</li><li>mean: 26.99 characters</li><li>max: 118 characters</li></ul> |
* Samples:
  | english                                                                 | non_english                                                    |
  |:------------------------------------------------------------------------|:---------------------------------------------------------------|
  | <code>Would you send a tomato juice, black coffee and a masseur?</code> | <code>هل لك أن ترسل لي عصير طماطم قهوة سوداء.. والمدلك!</code> |
  | <code>To hear the angels sing</code>                                    | <code>لكى تسمع غناء الملائكه</code>                            |
  | <code>Brace yourself.</code>                                            | <code>" تمالك نفسك " بريكر</code>                              |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>stackexchange</summary>

* Dataset: [stackexchange](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates) at [1c9657a](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates/tree/1c9657aec12d9e101667bb9593efcc623c4a68ff)
* Size: 250,519 training samples
* Columns: <code>post1</code> and <code>post2</code>
* Approximate statistics based on the first 1000 samples:
  |         | post1                                                                                             | post2                                                                                             |
  |:--------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
  | type    | string                                                                                            | string                                                                                            |
  | details | <ul><li>min: 77 characters</li><li>mean: 669.56 characters</li><li>max: 3982 characters</li></ul> | <ul><li>min: 81 characters</li><li>mean: 641.44 characters</li><li>max: 4053 characters</li></ul> |
* Samples:
  | post1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | post2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>New user question about passwords Just got a refurbished computer with Ubuntu as the OS.  Have never even heard of the OS and now I'm trying to learn.  When I boot the system, it starts up great.  But, if I try to navigate around, it requires a password.  Is there a trick to finding the initial password?  Please advise.</code>                                                                                                                                              | <code>How do I reset a lost administrative password? I'm working on a Ubuntu system, and my client has completely forgotten his administrative password. He doesn't even remember entering one; however it is there.  I've tried the suggestions on the website, and I have been unsuccessful in deleting the password so that I can download applets required for running some files. Is there a solution?</code>                                                                                                    |
  | <code>Reorder a list of string randomly but constant in a period of time I need to reorder a list in a random way but I want to have the same result on a short period of time ... So I have:  var list = new String[] { "Angie", "David", "Emily", "James" }    var shuffled = list.OrderBy(v =&gt; "4a78926c")).ToList();   But I always get the same order ... I could use Guid.NewGuid() but then I would have a different result in a short period of time.  How can I do this?</code> | <code>Randomize a List What is the best way to randomize the order of a generic list in C#? I've got a finite set of 75 numbers in a list I would like to assign a random order to, in order to draw them for a lottery type application.</code>                                                                                                                                                                                                                                                                      |
  | <code>Made a mistake on check need help to fix I wrote a check and put the amount in the pay to order spot. Can I just mark it out, put the name in the spot and finish writing the check?</code>                                                                                                                                                                                                                                                                                           | <code>How to correct a mistake made when writing a check? I think I know the answer to this, but I'm not sure, and it's a good question, so I'll ask:  What is the accepted/proper way to correct a mistake made on a check?  For instance, I imagine that in any given January, some people accidentally date a check in the previous year.  Is there a way to correct such a mistake, or must a check be voided (and wasted)?  Pointers to definitive information (U.S., Canada, and elsewhere) are helpful.</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>quora</summary>

* Dataset: [quora](https://huggingface.co/datasets/sentence-transformers/quora-duplicates) at [451a485](https://huggingface.co/datasets/sentence-transformers/quora-duplicates/tree/451a4850bd141edb44ade1b5828c259abd762cdb)
* Size: 101,762 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | positive                                                                                        | negative                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 16 characters</li><li>mean: 53.47 characters</li><li>max: 249 characters</li></ul> | <ul><li>min: 16 characters</li><li>mean: 52.63 characters</li><li>max: 237 characters</li></ul> | <ul><li>min: 14 characters</li><li>mean: 54.67 characters</li><li>max: 292 characters</li></ul> |
* Samples:
  | anchor                                                | positive                                         | negative                                            |
  |:------------------------------------------------------|:-------------------------------------------------|:----------------------------------------------------|
  | <code>What food should I try in Brazil?</code>        | <code>Which foods should I try in Brazil?</code> | <code>What meat should one eat in Argentina?</code> |
  | <code>What is the best way to get a threesome?</code> | <code>How does one find a threesome?</code>      | <code>How is the experience of a threesome?</code>  |
  | <code>Whether I do CA or MBA? Which is better?</code> | <code>Which is better CA or MBA?</code>          | <code>Which is better CA or IT?</code>              |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>wikianswers_duplicates</summary>

* Dataset: [wikianswers_duplicates](https://huggingface.co/datasets/sentence-transformers/wikianswers-duplicates) at [9af6367](https://huggingface.co/datasets/sentence-transformers/wikianswers-duplicates/tree/9af6367d1ad084daf8a9de9c21bc33fcdc7770d0)
* Size: 9,990,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | positive                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 14 characters</li><li>mean: 47.39 characters</li><li>max: 151 characters</li></ul> | <ul><li>min: 15 characters</li><li>mean: 47.58 characters</li><li>max: 154 characters</li></ul> |
* Samples:
  | anchor                                                                | positive                                                                 |
  |:----------------------------------------------------------------------|:-------------------------------------------------------------------------|
  | <code>Did Democritus belive matter was continess?</code>              | <code>Why did democritus call the smallest pice of matter atomos?</code> |
  | <code>Tell you about the most ever done to satisfy a customer?</code> | <code>How do you satisfy your client or customer?</code>                 |
  | <code>How is a chemical element different from a compound?</code>     | <code>How is a chemical element different to a compound?</code>          |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>all_nli</summary>

* Dataset: [all_nli](https://huggingface.co/datasets/sentence-transformers/all-nli) at [d482672](https://huggingface.co/datasets/sentence-transformers/all-nli/tree/d482672c8e74ce18da116f430137434ba2e52fab)
* Size: 557,850 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | positive                                                                                        | negative                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 18 characters</li><li>mean: 34.88 characters</li><li>max: 193 characters</li></ul> | <ul><li>min: 15 characters</li><li>mean: 46.49 characters</li><li>max: 181 characters</li></ul> | <ul><li>min: 16 characters</li><li>mean: 50.47 characters</li><li>max: 204 characters</li></ul> |
* Samples:
  | anchor                                                                     | positive                                         | negative                                                   |
  |:---------------------------------------------------------------------------|:-------------------------------------------------|:-----------------------------------------------------------|
  | <code>A person on a horse jumps over a broken down airplane.</code>        | <code>A person is outdoors, on a horse.</code>   | <code>A person is at a diner, ordering an omelette.</code> |
  | <code>Children smiling and waving at camera</code>                         | <code>There are children present</code>          | <code>The kids are frowning</code>                         |
  | <code>A boy is jumping on skateboard in the middle of a red bridge.</code> | <code>The boy does a skateboarding trick.</code> | <code>The boy skates down the sidewalk.</code>             |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>simple_wiki</summary>

* Dataset: [simple_wiki](https://huggingface.co/datasets/sentence-transformers/simple-wiki) at [60fd9b4](https://huggingface.co/datasets/sentence-transformers/simple-wiki/tree/60fd9b4680642ace0e2604cc2de44d376df419a7)
* Size: 102,225 training samples
* Columns: <code>text</code> and <code>simplified</code>
* Approximate statistics based on the first 1000 samples:
  |         | text                                                                                            | simplified                                                                                       |
  |:--------|:------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                           |
  | details | <ul><li>min: 18 characters</li><li>mean: 149.3 characters</li><li>max: 573 characters</li></ul> | <ul><li>min: 16 characters</li><li>mean: 123.58 characters</li><li>max: 576 characters</li></ul> |
* Samples:
  | text                                                                                                                                                                                                                              | simplified                                                                                                                                                                                                              |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>The next morning , it had a small CDO and well-defined bands , and the system , either a weak tropical storm or a strong tropical depression , likely reached its peak .</code>                                             | <code>The next morning , it had a small amounts of convection near the center and well-defined bands , and the system , either a weak tropical storm or a strong tropical depression , likely reached its peak .</code> |
  | <code>The region of measurable parameter space that corresponds to a regime is very often loosely defined . Examples include `` the superfluid regime '' , `` the steady state regime '' or `` the femtosecond regime '' .</code> | <code>This is common if a regime is threatened by another regime .</code>                                                                                                                                               |
  | <code>The Lamborghini Diablo is a high-performance mid-engined sports car that was built by Italian automaker Lamborghini between 1990 and 2001 .</code>                                                                          | <code>The Lamborghini Diablo is a sport car that was built by Lamborghini from 1990 to 2001 .</code>                                                                                                                    |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>altlex</summary>

* Dataset: [altlex](https://huggingface.co/datasets/sentence-transformers/altlex) at [97eb209](https://huggingface.co/datasets/sentence-transformers/altlex/tree/97eb20963455c361d5a81c107c3596cff9e0cd82)
* Size: 112,696 training samples
* Columns: <code>text</code> and <code>simplified</code>
* Approximate statistics based on the first 1000 samples:
  |         | text                                                                                             | simplified                                                                                       |
  |:--------|:-------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                           |
  | details | <ul><li>min: 13 characters</li><li>mean: 131.03 characters</li><li>max: 492 characters</li></ul> | <ul><li>min: 13 characters</li><li>mean: 112.41 characters</li><li>max: 492 characters</li></ul> |
* Samples:
  | text                                                                                                                                                                                                                                 | simplified                                                                                                                                           |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Reinforcement and punishment are the core tools of operant conditioning .</code>                                                                                                                                               | <code>Principles of operant conditioning :</code>                                                                                                    |
  | <code>The Japanese Ministry of Health , Labour and Welfare defines `` hikikomori '' as people who refuse to leave their house and , thus , isolate themselves from society in their homes for a period exceeding six months .</code> | <code>The Japanese Ministry of Health , Labour and Welfare defines hikikomori as people who refuse to leave their house for over six months .</code> |
  | <code>It has six rows of black spines and has a pair of long , clubbed spines on the head .</code>                                                                                                                                   | <code>It has a pair of long , clubbed spines on the head .</code>                                                                                    |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

#### flickr30k_captions

* Dataset: [flickr30k_captions](https://huggingface.co/datasets/sentence-transformers/flickr30k-captions) at [0ef0ce3](https://huggingface.co/datasets/sentence-transformers/flickr30k-captions/tree/0ef0ce31492fd8dc161ed483a40d3c4894f9a8c1)
* Size: 158,881 training samples
* Columns: <code>caption1</code> and <code>caption2</code>
* Approximate statistics based on the first 1000 samples:
  |         | caption1                                                                                        | caption2                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 20 characters</li><li>mean: 63.19 characters</li><li>max: 318 characters</li></ul> | <ul><li>min: 13 characters</li><li>mean: 63.65 characters</li><li>max: 205 characters</li></ul> |
* Samples:
  | caption1                                                                                          | caption2                                                                          |
  |:--------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | <code>Four women pose for a photograph with a man in a bright yellow suit.</code>                 | <code>A group of friends get their photo taken with a man in a green suit.</code> |
  | <code>A many dressed in army gear walks on the crash walking a brown dog.</code>                  | <code>A man with army fatigues is walking his dog.</code>                         |
  | <code>Four people are sitting around a kitchen counter while one is drinking from a glass.</code> | <code>A group of people sit around a breakfast bar.</code>                        |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>coco_captions</summary>

* Dataset: [coco_captions](https://huggingface.co/datasets/sentence-transformers/coco-captions) at [bd26018](https://huggingface.co/datasets/sentence-transformers/coco-captions/tree/bd2601822b9af9a41656d678ffbd5c80d81e276a)
* Size: 414,010 training samples
* Columns: <code>caption1</code> and <code>caption2</code>
* Approximate statistics based on the first 1000 samples:
  |         | caption1                                                                                        | caption2                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 30 characters</li><li>mean: 52.57 characters</li><li>max: 151 characters</li></ul> | <ul><li>min: 29 characters</li><li>mean: 52.71 characters</li><li>max: 186 characters</li></ul> |
* Samples:
  | caption1                                                                                                     | caption2                                                                             |
  |:-------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | <code>THERE ARE FRIENDS ON THE BEACH POSING </code>                                                          | <code>A group of people standing together on the beach while holding a woman.</code> |
  | <code>a lovely white bathroom with white shower curtain.</code>                                              | <code>A white toilet sitting in a bathroom next to a sink.</code>                    |
  | <code>Two drinking glass on a counter and a man holding a knife looking at something in front of him.</code> | <code>A restaurant employee standing behind two cups on a counter.</code>            |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>nli_for_simcse</summary>

* Dataset: [nli_for_simcse](https://huggingface.co/datasets/sentence-transformers/nli-for-simcse) at [926cae4](https://huggingface.co/datasets/sentence-transformers/nli-for-simcse/tree/926cae4af15a99b5cc2b053212bb52a4b377c418)
* Size: 274,951 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | positive                                                                                       | negative                                                                                       |
  |:--------|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 11 characters</li><li>mean: 87.69 characters</li><li>max: 483 characters</li></ul> | <ul><li>min: 7 characters</li><li>mean: 43.85 characters</li><li>max: 244 characters</li></ul> | <ul><li>min: 7 characters</li><li>mean: 43.87 characters</li><li>max: 172 characters</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                       | positive                                                              | negative                                                   |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------|:-----------------------------------------------------------|
  | <code>A white horse and a rider wearing a ale blue shirt, white pants, and a black helmet are jumping a hurdle.</code>                                                                                                       | <code>An equestrian is having a horse jump a hurdle.</code>           | <code>A competition is taking place in a kitchen.</code>   |
  | <code>A group of people in a dome like building.</code>                                                                                                                                                                      | <code>A gathering inside a building.</code>                           | <code>Cats are having a party.</code>                      |
  | <code>Home to thousands of sheep and a few scattered farming families, the area is characterized by the stark beauty of bare peaks, rugged fells, and the most remote lakes, combined with challenging, narrow roads.</code> | <code>There are no wide and easy roads going through the area.</code> | <code>There are more humans than sheep in the area.</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>negation</summary>

* Dataset: [negation](https://huggingface.co/datasets/jinaai/negation-dataset) at [cd02256](https://huggingface.co/datasets/jinaai/negation-dataset/tree/cd02256426cc566d176285a987e5436f1cd01382)
* Size: 10,000 training samples
* Columns: <code>anchor</code>, <code>entailment</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                         | entailment                                                                                     | negative                                                                                       |
  |:--------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 9 characters</li><li>mean: 65.84 characters</li><li>max: 275 characters</li></ul> | <ul><li>min: 7 characters</li><li>mean: 34.06 characters</li><li>max: 167 characters</li></ul> | <ul><li>min: 9 characters</li><li>mean: 37.26 characters</li><li>max: 166 characters</li></ul> |
* Samples:
  | anchor                                                                                             | entailment                                                        | negative                                                              |
  |:---------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|:----------------------------------------------------------------------|
  | <code>A boy with his hands above his head stands on a cement pillar above the cobblestones.</code> | <code>A boy is standing on a pillar over the cobblestones.</code> | <code>A boy is not standing on a pillar over the cobblestones.</code> |
  | <code>The man works hard in his home office.</code>                                                | <code>home based worker works harder</code>                       | <code>home based worker does not work harder</code>                   |
  | <code>Man in black shirt plays silver electric guitar.</code>                                      | <code>A man plays a silver electric guitar.</code>                | <code>A man does not play a silver electric guitar.</code>            |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

### Evaluation Datasets

<details><summary>wikititles</summary>

* Dataset: [wikititles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikititles) at [d92a4d2](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikititles/tree/d92a4d28a082c3c93563feb92a77de6074bdeb52)
* Size: 14,700,458 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                       | non_english                                                                                  |
  |:--------|:----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
  | type    | string                                                                                        | string                                                                                       |
  | details | <ul><li>min: 4 characters</li><li>mean: 18.33 characters</li><li>max: 77 characters</li></ul> | <ul><li>min: 4 characters</li><li>mean: 17.3 characters</li><li>max: 83 characters</li></ul> |
* Samples:
  | english                                                          | non_english                          |
  |:-----------------------------------------------------------------|:-------------------------------------|
  | <code>Bjørvika</code>                                            | <code>比約維卡</code>                    |
  | <code>Old Mystic, Connecticut</code>                             | <code>Олд Мистик (Конектикат)</code> |
  | <code>Cystic fibrosis transmembrane conductance regulator</code> | <code>CFTR</code>                    |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>tatoeba</summary>

* Dataset: [tatoeba](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba) at [cec1343](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba/tree/cec1343ab5a7a8befe99af4a2d0ca847b6c84743)
* Size: 4,138,956 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                        | non_english                                                                                   |
  |:--------|:-----------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                        |
  | details | <ul><li>min: 5 characters</li><li>mean: 31.83 characters</li><li>max: 235 characters</li></ul> | <ul><li>min: 4 characters</li><li>mean: 31.7 characters</li><li>max: 189 characters</li></ul> |
* Samples:
  | english                                              | non_english                                          |
  |:-----------------------------------------------------|:-----------------------------------------------------|
  | <code>You are not consistent in your actions.</code> | <code>Je bent niet consequent in je handelen.</code> |
  | <code>Neither of them seemed old.</code>             | <code>Ninguno de ellos lucía viejo.</code>           |
  | <code>Stand up, please.</code>                       | <code>Устаните, молим Вас.</code>                    |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>talks</summary>

* Dataset: [talks](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks) at [0c70bc6](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks/tree/0c70bc6714efb1df12f8a16b9056e4653563d128)
* Size: 9,750,031 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                        | non_english                                                                                    |
  |:--------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 9 characters</li><li>mean: 94.78 characters</li><li>max: 634 characters</li></ul> | <ul><li>min: 4 characters</li><li>mean: 84.61 characters</li><li>max: 596 characters</li></ul> |
* Samples:
  | english                                                           | non_english                                                            |
  |:------------------------------------------------------------------|:-----------------------------------------------------------------------|
  | <code>I'm earthed in my essence, and my self is suspended.</code> | <code>Je suis ancrée, et mon moi est temporairement inexistant.</code> |
  | <code>It's not back on your shoulder.</code>                      | <code>Dar nu e înapoi pe umăr.</code>                                  |
  | <code>They're usually students who've never seen a desert.</code> | <code>たいていの学生は砂漠を見たこともありません</code>                                     |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>europarl</summary>

* Dataset: [europarl](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-europarl) at [11007ec](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-europarl/tree/11007ecf9c790178a49a4cbd5cfea451a170f2dc)
* Size: 10,000 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                          | non_english                                                                                      |
  |:--------|:-------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                           |
  | details | <ul><li>min: 0 characters</li><li>mean: 148.52 characters</li><li>max: 1215 characters</li></ul> | <ul><li>min: 0 characters</li><li>mean: 154.44 characters</li><li>max: 1316 characters</li></ul> |
* Samples:
  | english                                                                                                                                                                                                                                                   | non_english                                                                                                                                                                                                                                                                                                                |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Mr Schmidt, Mr Trichet, I absolutely cannot go along with these proposals.</code>                                                                                                                                                                   | <code>Pane Schmidte, pane Trichete, s těmito návrhy nemohu vůbec souhlasit.</code>                                                                                                                                                                                                                                         |
  | <code>The Council and Parliament recently adopted the regulation on the Single European Sky, one of the provisions of which was Community membership of Eurocontrol, so that Parliament has already indirectly expressed its views on this matter.</code> | <code>Der Rat und das Parlament haben kürzlich die Verordnung über die Schaffung eines einheitlichen europäischen Luftraums verabschiedet, in der unter anderem die Mitgliedschaft der Gemeinschaft bei Eurocontrol festgelegt ist, so dass das Parlament seine Auffassungen hierzu indirekt bereits dargelegt hat.</code> |
  | <code>It was held over from the January part-session until this part-session.</code>                                                                                                                                                                      | <code>Ihre Behandlung wurde von der Januar-Sitzung auf die jetzige vertagt.</code>                                                                                                                                                                                                                                         |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>global_voices</summary>

* Dataset: [global_voices](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices) at [4cc20ad](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices/tree/4cc20add371f246bb1559b543f8b0dea178a1803)
* Size: 1,099,099 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                         | non_english                                                                                     |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 3 characters</li><li>mean: 115.61 characters</li><li>max: 629 characters</li></ul> | <ul><li>min: 3 characters</li><li>mean: 121.61 characters</li><li>max: 664 characters</li></ul> |
* Samples:
  | english                                                                                                                                                  | non_english                                                                                                                                                           |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Haiti: Security vs. Relief? · Global Voices</code>                                                                                                 | <code>Haïti : Zones rouges, zones vertes - sécurité contre aide humanitaire ?</code>                                                                                  |
  | <code>In order to prevent weapon smuggling through tunnels, his forces would have fought and killed Palestinians over a sustained period of time.</code> | <code>Con el fin de impedir el contrabando de armas a través de túneles, sus fuerzas habrían combatido y muerto palestinos durante un largo período de tiempo.</code> |
  | <code>Tombstone of Vitalis, an ancient Roman cavalry officer, displayed in front of the Skopje City Museum.</code>                                       | <code>Lápida de Vitalis, un antiguo oficial romano de caballería, exhibida frente al Museo de la Ciudad de Skopje.</code>                                             |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>muse</summary>

* Dataset: [muse](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-muse) at [238c077](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-muse/tree/238c077ac66070748aaf2ab1e45185b0145b7291)
* Size: 1,368,274 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                     | non_english                                                                                  |
  |:--------|:--------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
  | type    | string                                                                                      | string                                                                                       |
  | details | <ul><li>min: 3 characters</li><li>mean: 7.5 characters</li><li>max: 17 characters</li></ul> | <ul><li>min: 1 characters</li><li>mean: 7.39 characters</li><li>max: 16 characters</li></ul> |
* Samples:
  | english                  | non_english              |
  |:-------------------------|:-------------------------|
  | <code>generalised</code> | <code>γενικευμένη</code> |
  | <code>language</code>    | <code>jazyku</code>      |
  | <code>finalised</code>   | <code>финализиран</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>wikimatrix</summary>

* Dataset: [wikimatrix](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix) at [74a4cb1](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix/tree/74a4cb15422cdd0c3aacc93593b6cb96a9b9b3a9)
* Size: 9,688,498 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                         | non_english                                                                                      |
  |:--------|:------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                           |
  | details | <ul><li>min: 11 characters</li><li>mean: 122.6 characters</li><li>max: 424 characters</li></ul> | <ul><li>min: 10 characters</li><li>mean: 128.09 characters</li><li>max: 579 characters</li></ul> |
* Samples:
  | english                                                                                                            | non_english                                                                                                                      |
  |:-------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
  | <code>Along with the adjacent waters, it was declared a nature reserve in 2002.</code>                             | <code>Juntament amb les aigües adjacents, va ser declarada reserva natural el 2002.</code>                                       |
  | <code>Like her husband, Charlotte was a patron of astronomy.</code>                                                | <code>Stejně jako manžel byla Šarlota patronkou astronomie.</code>                                                               |
  | <code>Some of the music consists of simple sounds, such as a wind effect heard over the poem "Soon Alaska".</code> | <code>Sommige muziekstukken bevatten eenvoudige geluiden, zoals het geluid van de wind tijdens het gedicht "Soon Alaska".</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>opensubtitles</summary>

* Dataset: [opensubtitles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles) at [d86a387](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles/tree/d86a387587ab6f2fd9ec7453b2765cec68111c87)
* Size: 10,000 evaluation samples
* Columns: <code>english</code> and <code>non_english</code>
* Approximate statistics based on the first 1000 samples:
  |         | english                                                                                        | non_english                                                                                    |
  |:--------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 0 characters</li><li>mean: 35.01 characters</li><li>max: 200 characters</li></ul> | <ul><li>min: 0 characters</li><li>mean: 27.79 characters</li><li>max: 143 characters</li></ul> |
* Samples:
  | english                                    | non_english                            |
  |:-------------------------------------------|:---------------------------------------|
  | <code>- I don't need my medicine.</code>   | <code>-لا أحتاج لدوائي</code>          |
  | <code>The Sovereign... Ah.</code>          | <code>(الطاغية)!</code>                |
  | <code>The other two from your ship.</code> | <code>الإثنان الأخران من سفينتك</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>stackexchange</summary>

* Dataset: [stackexchange](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates) at [1c9657a](https://huggingface.co/datasets/sentence-transformers/stackexchange-duplicates/tree/1c9657aec12d9e101667bb9593efcc623c4a68ff)
* Size: 250,519 evaluation samples
* Columns: <code>post1</code> and <code>post2</code>
* Approximate statistics based on the first 1000 samples:
  |         | post1                                                                                             | post2                                                                                             |
  |:--------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
  | type    | string                                                                                            | string                                                                                            |
  | details | <ul><li>min: 64 characters</li><li>mean: 669.92 characters</li><li>max: 4103 characters</li></ul> | <ul><li>min: 62 characters</li><li>mean: 644.68 characters</li><li>max: 4121 characters</li></ul> |
* Samples:
  | post1                                                                                                                                                                                                                                                                                                                                                                       | post2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Find the particular solution for this linear ODE $y' '-2y'+5y=e^x \cos2x$. Find the particular solution for this linear ODE :$y' '-2y'+5y=e^x \cos2x$.  How can I use Undetermined coefficients method ?</code>                                                                                                                                                       | <code>Particular solution of $y''-4y'+5y = 4e^{2x} (\sin x)$ How do I find the particular solution of this second order inhomogenous differential equation? (Using undetermined coefficients).  $y''-4y'+5y = 4e^{2x} (\sin x)$  I can find the generel homogenous solutions but I need help for the particular.</code>                                                                                                                                                                                                                                                       |
  | <code>Unbounded sequence has an divergent subsequence Show that if $(x_n)$ is unbounded, then there exists a subsequence $(x_{n_k})$ such that $\lim 1/(x_{n_k}) =0.$ I was thinking that $(x_n)$ is a subsequence of itself. WLOG, suppose $(x_n)$ does not have an upper bound. By Algebraic Limit Theorem, $\lim 1/(x_{n_k}) =0.$ Is there any flaws in my proof?</code> | <code>Given the sequence $(x_n)$ is unbounded, show that there exist a subsequence $(x_{n_k})$ such that $\lim(1/x_n)=0$. Given the sequence $(x_n)$ is unbounded, show that there exist a subsequence $(x_{n_k})$ such that $\lim(1/x_{n_k})=0$.   I guess I have to prove that $(x_{n_k})$ diverge, but I don't know how to carry on.  Thanks.</code>                                                                                                                                                                                                                       |
  | <code>"The problem is who can we get to replace her" vs. "The problem is who we can get to replace her" "The problem is who can we get to replace her" vs. "The problem is who we can get to replace her" Which one is correct and why?</code>                                                                                                                              | <code>Changing subject and verb positions in statements and questions We always change subject and verb positions in whenever we want to ask a question such as "What is your name?". But when it comes to statements like the following, which form is correct?        I don't understand what are you talking about.   I don't understand what you are talking about.      Another example        Do you know what time is it?   Do you know what time it is?      Another example        Do you care how do I feel about this?   Do you care how I feel about this?</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>quora</summary>

* Dataset: [quora](https://huggingface.co/datasets/sentence-transformers/quora-duplicates) at [451a485](https://huggingface.co/datasets/sentence-transformers/quora-duplicates/tree/451a4850bd141edb44ade1b5828c259abd762cdb)
* Size: 101,762 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | positive                                                                                        | negative                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 15 characters</li><li>mean: 52.48 characters</li><li>max: 164 characters</li></ul> | <ul><li>min: 12 characters</li><li>mean: 52.86 characters</li><li>max: 162 characters</li></ul> | <ul><li>min: 12 characters</li><li>mean: 56.18 characters</li><li>max: 298 characters</li></ul> |
* Samples:
  | anchor                                                | positive                                                   | negative                                                                                                                                                                                                          |
  |:------------------------------------------------------|:-----------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Is pornography an art?</code>                   | <code>Can pornography be art?</code>                       | <code>Does pornography involve the objectification of women?</code>                                                                                                                                               |
  | <code>How can I improve my speaking in public?</code> | <code>How can I improve my public speaking ability?</code> | <code>How do I improve my vocabulary and English speaking skills? I am a 22 year old software engineer and come from a Telugu medium background. I am able to write well, but my speaking skills are poor.</code> |
  | <code>How do I develop better people skills?</code>   | <code>How can I get better people skills?</code>           | <code>How do I get better at Minecraft?</code>                                                                                                                                                                    |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>wikianswers_duplicates</summary>

* Dataset: [wikianswers_duplicates](https://huggingface.co/datasets/sentence-transformers/wikianswers-duplicates) at [9af6367](https://huggingface.co/datasets/sentence-transformers/wikianswers-duplicates/tree/9af6367d1ad084daf8a9de9c21bc33fcdc7770d0)
* Size: 10,000 evaluation samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | positive                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 14 characters</li><li>mean: 47.88 characters</li><li>max: 145 characters</li></ul> | <ul><li>min: 15 characters</li><li>mean: 47.76 characters</li><li>max: 201 characters</li></ul> |
* Samples:
  | anchor                                                                    | positive                                                     |
  |:--------------------------------------------------------------------------|:-------------------------------------------------------------|
  | <code>Can you get pregnant if tubes are clamped?</code>                   | <code>How long can your fallopian tubes stay clamped?</code> |
  | <code>Is there any object that are triangular prism?</code>               | <code>Is a trapezium the same as a triangular prism?</code>  |
  | <code>Where is the neutral switch located on a 2000 ford explorer?</code> | <code>Ford f150 1996 safety switch?</code>                   |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>all_nli</summary>

* Dataset: [all_nli](https://huggingface.co/datasets/sentence-transformers/all-nli) at [d482672](https://huggingface.co/datasets/sentence-transformers/all-nli/tree/d482672c8e74ce18da116f430137434ba2e52fab)
* Size: 6,584 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | positive                                                                                        | negative                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 15 characters</li><li>mean: 72.82 characters</li><li>max: 300 characters</li></ul> | <ul><li>min: 12 characters</li><li>mean: 34.11 characters</li><li>max: 126 characters</li></ul> | <ul><li>min: 11 characters</li><li>mean: 36.38 characters</li><li>max: 121 characters</li></ul> |
* Samples:
  | anchor                                                                                                                                                                         | positive                                                    | negative                                                |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:--------------------------------------------------------|
  | <code>Two women are embracing while holding to go packages.</code>                                                                                                             | <code>Two woman are holding packages.</code>                | <code>The men are fighting outside a deli.</code>       |
  | <code>Two young children in blue jerseys, one with the number 9 and one with the number 2 are standing on wooden steps in a bathroom and washing their hands in a sink.</code> | <code>Two kids in numbered jerseys wash their hands.</code> | <code>Two kids in jackets walk to school.</code>        |
  | <code>A man selling donuts to a customer during a world exhibition event held in the city of Angeles</code>                                                                    | <code>A man selling donuts to a customer.</code>            | <code>A woman drinks her coffee in a small cafe.</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>simple_wiki</summary>

* Dataset: [simple_wiki](https://huggingface.co/datasets/sentence-transformers/simple-wiki) at [60fd9b4](https://huggingface.co/datasets/sentence-transformers/simple-wiki/tree/60fd9b4680642ace0e2604cc2de44d376df419a7)
* Size: 102,225 evaluation samples
* Columns: <code>text</code> and <code>simplified</code>
* Approximate statistics based on the first 1000 samples:
  |         | text                                                                                             | simplified                                                                                       |
  |:--------|:-------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                           |
  | details | <ul><li>min: 24 characters</li><li>mean: 147.36 characters</li><li>max: 599 characters</li></ul> | <ul><li>min: 19 characters</li><li>mean: 124.94 characters</li><li>max: 540 characters</li></ul> |
* Samples:
  | text                                                                                                                                                                                 | simplified                                                                                                              |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
  | <code>It marks the southernmost point of the BahÃ a de Banderas , upon which the port and resort city of Puerto Vallarta stands .</code>                                             | <code>It is the most southern point of the BahÃ a de Banderas .</code>                                                  |
  | <code>The interiors of the stations resemble that of the former western Soviet nations , with chandeliers hanging from the corridors .</code>                                        | <code>Its interior resembles that of western former Soviet nations with chandeliers hanging from the corridors .</code> |
  | <code>The Senegal national football team , nicknamed the Lions of Teranga , is the national team of Senegal and is controlled by the FÃ dÃ ration SÃ nÃ galaise de Football .</code> | <code>Senegal national football team is the national football team of Senegal .</code>                                  |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>altlex</summary>

* Dataset: [altlex](https://huggingface.co/datasets/sentence-transformers/altlex) at [97eb209](https://huggingface.co/datasets/sentence-transformers/altlex/tree/97eb20963455c361d5a81c107c3596cff9e0cd82)
* Size: 112,696 evaluation samples
* Columns: <code>text</code> and <code>simplified</code>
* Approximate statistics based on the first 1000 samples:
  |         | text                                                                                            | simplified                                                                                      |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 9 characters</li><li>mean: 138.99 characters</li><li>max: 592 characters</li></ul> | <ul><li>min: 7 characters</li><li>mean: 119.43 characters</li><li>max: 517 characters</li></ul> |
* Samples:
  | text                                                                                                   | simplified                                                                      |
  |:-------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|
  | <code>14,000 ) referred to as `` The bush '' within the media .</code>                                 | <code>14,000 ) called `` the bush '' in the media .</code>                      |
  | <code>The next day he told Elizabeth everything he knew regarding Catherine and her pregnancy .</code> | <code>The next day he told Elizabeth everything .</code>                        |
  | <code>Alice Ivers and Warren Tubbs had four sons and three daughters together .</code>                 | <code>Alice Ivers and Warren Tubbs had 4 sons and 3 daughters together .</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>flickr30k_captions</summary>

* Dataset: [flickr30k_captions](https://huggingface.co/datasets/sentence-transformers/flickr30k-captions) at [0ef0ce3](https://huggingface.co/datasets/sentence-transformers/flickr30k-captions/tree/0ef0ce31492fd8dc161ed483a40d3c4894f9a8c1)
* Size: 158,881 evaluation samples
* Columns: <code>caption1</code> and <code>caption2</code>
* Approximate statistics based on the first 1000 samples:
  |         | caption1                                                                                        | caption2                                                                                        |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          |
  | details | <ul><li>min: 12 characters</li><li>mean: 62.95 characters</li><li>max: 279 characters</li></ul> | <ul><li>min: 15 characters</li><li>mean: 63.34 characters</li><li>max: 206 characters</li></ul> |
* Samples:
  | caption1                                                                                              | caption2                                                                                   |
  |:------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------|
  | <code>A person wearing sunglasses, a visor, and a British flag is carrying 6 Heineken bottles.</code> | <code>A woman wearing a blue visor is holding 5 bottles of Heineken beer.</code>           |
  | <code>Two older people hold hands while walking down a street alley with a group of people.</code>    | <code>A group of senior citizens walking down narrow pathway.</code>                       |
  | <code>View of bicyclists from behind during a race.</code>                                            | <code>A Peloton of bicyclists riding down a road of tightly packed together houses.</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>coco_captions</summary>

* Dataset: [coco_captions](https://huggingface.co/datasets/sentence-transformers/coco-captions) at [bd26018](https://huggingface.co/datasets/sentence-transformers/coco-captions/tree/bd2601822b9af9a41656d678ffbd5c80d81e276a)
* Size: 414,010 evaluation samples
* Columns: <code>caption1</code> and <code>caption2</code>
* Approximate statistics based on the first 1000 samples:
  |         | caption1                                                                                       | caption2                                                                                       |
  |:--------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 26 characters</li><li>mean: 51.9 characters</li><li>max: 130 characters</li></ul> | <ul><li>min: 28 characters</li><li>mean: 52.7 characters</li><li>max: 135 characters</li></ul> |
* Samples:
  | caption1                                                          | caption2                                                              |
  |:------------------------------------------------------------------|:----------------------------------------------------------------------|
  | <code>A blurry photo of a man next to a refrigerator</code>       | <code>The man in black is moving towards a refrigerator.</code>       |
  | <code>A young child holding a remote control in it's hand.</code> | <code>A boy holds a remote control up to the camera.</code>           |
  | <code>a big airplane that is parked on some concrete</code>       | <code>A man standing next to a fighter jet under a cloudy sky.</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>nli_for_simcse</summary>

* Dataset: [nli_for_simcse](https://huggingface.co/datasets/sentence-transformers/nli-for-simcse) at [926cae4](https://huggingface.co/datasets/sentence-transformers/nli-for-simcse/tree/926cae4af15a99b5cc2b053212bb52a4b377c418)
* Size: 274,951 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                         | positive                                                                                        | negative                                                                                       |
  |:--------|:-----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                          | string                                                                                         |
  | details | <ul><li>min: 9 characters</li><li>mean: 84.79 characters</li><li>max: 598 characters</li></ul> | <ul><li>min: 10 characters</li><li>mean: 44.26 characters</li><li>max: 172 characters</li></ul> | <ul><li>min: 9 characters</li><li>mean: 44.11 characters</li><li>max: 134 characters</li></ul> |
* Samples:
  | anchor                                                                                                                                  | positive                                                                                           | negative                                                           |
  |:----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
  | <code>a man waiting for train with a blue coat blue jeans while holing a rope.</code>                                                   | <code>A man is waiting for a train.</code>                                                         | <code>A man is sitting on a greyhound bus waiting to leave.</code> |
  | <code>Australia's floating dollar has apparently allowed the island continent to sail almost unscathed through the Asian crisis.</code> | <code>Australia has a floating dollar that has made them impervious to the problem in Asia.</code> | <code>Australia has a dollar that is heavily tied to Asia.</code>  |
  | <code>A city street in front of a business with a construction worker and road cones.</code>                                            | <code>There is a city street with construction worker and road cones.</code>                       | <code>There are no cones in front of the city street.</code>       |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

</details>

<details><summary>negation</summary>

* Dataset: [negation](https://huggingface.co/datasets/jinaai/negation-dataset) at [cd02256](https://huggingface.co/datasets/jinaai/negation-dataset/tree/cd02256426cc566d176285a987e5436f1cd01382)
* Size: 10,000 evaluation samples
* Columns: <code>anchor</code>, <code>entailment</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                                          | entailment                                                                                     | negative                                                                                       |
  |:--------|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                         | string                                                                                         |
  | details | <ul><li>min: 26 characters</li><li>mean: 69.49 characters</li><li>max: 229 characters</li></ul> | <ul><li>min: 15 characters</li><li>mean: 34.88 characters</li><li>max: 89 characters</li></ul> | <ul><li>min: 16 characters</li><li>mean: 38.68 characters</li><li>max: 87 characters</li></ul> |
* Samples:
  | anchor                                                                                                                                       | entailment                                                                | negative                                                                          |
  |:---------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | <code>Two men, one standing and one seated on the ground are attempting to wrangle a bull as dust from the action is being kicked up.</code> | <code>Two cowboys attempt to wrangle a bull.</code>                       | <code>Two cowboys do not attempt to wrangle a bull.</code>                        |
  | <code>A woman dressed in black is silhouetted against a cloud darkened sky.</code>                                                           | <code>A woman in black stands in front of a dark, cloudy backdrop.</code> | <code>A woman in black does not stand in front of a dark, cloudy backdrop.</code> |
  | <code>A kid in a blue shirt playing on a playground.</code>                                                                                  | <code>A kid playing on a playground wearing a blue shirt</code>           | <code>A kid not playing on a playground wearing a black shirt</code>              |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          1024,
          512,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```
</details>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 2048
- `per_device_eval_batch_size`: 2048
- `learning_rate`: 0.2
- `num_train_epochs`: 1
- `warmup_ratio`: 0.1
- `bf16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 2048
- `per_device_eval_batch_size`: 2048
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 0.2
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: True
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step  | Training Loss | wikititles loss | tatoeba loss | talks loss | europarl loss | global voices loss | muse loss | wikimatrix loss | opensubtitles loss | stackexchange loss | quora loss | wikianswers duplicates loss | all nli loss | simple wiki loss | altlex loss | flickr30k captions loss | coco captions loss | nli for simcse loss | negation loss |
|:------:|:-----:|:-------------:|:---------------:|:------------:|:----------:|:-------------:|:------------------:|:---------:|:---------------:|:------------------:|:------------------:|:----------:|:---------------------------:|:------------:|:----------------:|:-----------:|:-----------------------:|:------------------:|:-------------------:|:-------------:|
| 0.0000 | 1     | 38.504        | -               | -            | -          | -             | -                  | -         | -               | -                  | -                  | -          | -                           | -            | -                | -           | -                       | -                  | -                   | -             |
| 0.0327 | 1000  | 21.3661       | 15.2607         | 9.1892       | 11.6736    | 1.6431        | 6.6894             | 31.9579   | 3.0122          | 0.3541             | 5.1814             | 2.3756     | 4.9474                      | 12.7699      | 0.5687           | 0.8911      | 21.0068                 | 17.1302            | 10.8964             | 6.7603        |
| 0.0654 | 2000  | 9.8377        | 11.7637         | 7.1680       | 8.7697     | 1.6077        | 5.2310             | 27.4887   | 1.8375          | 0.3379             | 5.1107             | 2.2083     | 4.1690                      | 12.0384      | 0.4837           | 0.7131      | 20.5401                 | 17.8388            | 10.6706             | 7.0488        |
| 0.0982 | 3000  | 8.5279        | 10.8719         | 6.6160       | 8.3116     | 1.5638        | 4.7298             | 25.8572   | 1.6738          | 0.3152             | 5.1009             | 2.0893     | 3.7332                      | 12.0452      | 0.4285           | 0.6519      | 20.2154                 | 16.2715            | 10.7693             | 7.3144        |
| 0.1309 | 4000  | 7.8208        | 10.4614         | 5.4918       | 7.4421     | 1.4420        | 4.0505             | 24.9000   | 1.3462          | 0.2925             | 4.7643             | 2.1143     | 3.7457                      | 11.6570      | 0.4390           | 0.6536      | 19.4405                 | 16.0912            | 10.7537             | 7.2120        |
| 0.1636 | 5000  | 7.5347        | 9.5381          | 5.9489       | 7.4027     | 1.4858        | 4.0272             | 23.8335   | 1.2453          | 0.3027             | 3.1262             | 1.9170     | 3.7535                      | 11.6186      | 0.4090           | 0.6131      | 18.9329                 | 16.1769            | 10.1123             | 7.0750        |
| 0.1963 | 6000  | 7.1819        | 9.2175          | 5.3231       | 7.0836     | 1.4795        | 3.8328             | 23.1620   | 1.1609          | 0.2964             | 2.7653             | 1.9440     | 3.6610                      | 11.2147      | 0.3714           | 0.5853      | 19.0478                 | 16.4413            | 9.5790              | 6.8695        |
| 0.2291 | 7000  | 6.9852        | 9.0344          | 5.5773       | 6.7928     | 1.4409        | 3.9232             | 23.2098   | 1.1750          | 0.2877             | 2.9254             | 1.9411     | 3.5469                      | 11.0744      | 0.4254           | 0.6293      | 19.0447                 | 16.3774            | 9.5363              | 6.8393        |
| 0.2618 | 8000  | 6.8114        | 8.9620          | 5.1417       | 6.5466     | 1.4834        | 3.7100             | 22.9815   | 1.0679          | 0.2942             | 2.7687             | 2.0211     | 3.6063                      | 11.3424      | 0.4447           | 0.6223      | 19.1836                 | 16.5669            | 9.8785              | 6.8528        |
| 0.2945 | 9000  | 6.5487        | 8.6320          | 4.8710       | 6.5144     | 1.4156        | 3.5712             | 22.9660   | 1.0261          | 0.3051             | 3.0898             | 1.9981     | 3.4305                      | 11.1448      | 0.3729           | 0.5814      | 18.8865                 | 15.8581            | 9.5213              | 6.7567        |
| 0.3272 | 10000 | 6.7398        | 8.5630          | 4.7179       | 6.5025     | 1.3931        | 3.5699             | 22.5319   | 0.9916          | 0.2870             | 3.3385             | 1.9580     | 3.5807                      | 11.2592      | 0.4155           | 0.6009      | 19.1387                 | 16.6836            | 9.6300              | 6.6613        |
| 0.3599 | 11000 | 6.3915        | 8.4041          | 4.8985       | 6.2787     | 1.4081        | 3.5082             | 22.3204   | 0.9554          | 0.2916             | 2.9365             | 2.0176     | 3.3900                      | 11.2956      | 0.3902           | 0.5783      | 18.6448                 | 16.1241            | 9.5388              | 6.7295        |
| 0.3927 | 12000 | 6.5902        | 8.1888          | 4.7326       | 6.1930     | 1.4550        | 3.4999             | 22.1070   | 0.9736          | 0.2935             | 2.9612             | 1.9449     | 3.3281                      | 11.0477      | 0.3821           | 0.5696      | 18.3227                 | 16.1848            | 9.4772              | 7.0029        |
| 0.4254 | 13000 | 6.341         | 8.1827          | 4.3838       | 6.1052     | 1.4165        | 3.3944             | 21.9552   | 0.9076          | 0.2991             | 3.2272             | 1.9822     | 3.3494                      | 11.1891      | 0.3790           | 0.5600      | 18.4394                 | 15.9000            | 9.5644              | 6.9056        |
| 0.4581 | 14000 | 6.2067        | 8.1549          | 4.4833       | 6.0765     | 1.4055        | 3.3903             | 21.4785   | 0.8962          | 0.2919             | 2.8893             | 1.9540     | 3.3078                      | 11.2100      | 0.3569           | 0.5461      | 18.7667                 | 16.2978            | 9.2310              | 7.1290        |
| 0.4908 | 15000 | 6.2237        | 8.0711          | 4.4755       | 6.0087     | 1.3185        | 3.2888             | 21.3689   | 0.8433          | 0.2861             | 3.0129             | 1.9084     | 3.3279                      | 11.1236      | 0.3730           | 0.5553      | 18.2711                 | 15.7648            | 9.5295              | 7.0092        |
| 0.5236 | 16000 | 6.1058        | 8.0282          | 4.5076       | 5.8760     | 1.4234        | 3.3046             | 21.3568   | 0.8298          | 0.2826             | 2.8404             | 1.8920     | 3.2918                      | 11.1140      | 0.3811           | 0.5550      | 18.2899                 | 15.8630            | 9.4807              | 6.7585        |
| 0.5563 | 17000 | 6.3038        | 7.8679          | 4.4780       | 5.8461     | 1.4016        | 3.2279             | 21.0624   | 0.8205          | 0.2804             | 3.1359             | 1.9066     | 3.3205                      | 11.0882      | 0.3913           | 0.5569      | 18.0693                 | 15.7346            | 9.2854              | 6.9239        |
| 0.5890 | 18000 | 5.9824        | 7.7827          | 4.3199       | 5.7441     | 1.3582        | 3.1982             | 21.2444   | 0.8046          | 0.2797             | 2.7466             | 1.8717     | 3.3112                      | 11.0553      | 0.3922           | 0.5568      | 18.0357                 | 15.6732            | 9.6404              | 6.8331        |
| 0.6217 | 19000 | 6.0275        | 7.7201          | 4.3591       | 5.8132     | 1.3466        | 3.1888             | 20.9311   | 0.8019          | 0.2765             | 2.7674             | 1.8670     | 3.3082                      | 10.9725      | 0.3996           | 0.5560      | 18.6346                 | 16.2965            | 9.3774              | 6.9957        |
| 0.6545 | 20000 | 6.1161        | 7.6429          | 4.2702       | 5.7298     | 1.3670        | 3.1433             | 20.8899   | 0.7871          | 0.2761             | 2.7486             | 1.9230     | 3.2958                      | 11.0207      | 0.3516           | 0.5361      | 18.2297                 | 15.6363            | 9.6376              | 7.1608        |
| 0.6872 | 21000 | 5.9608        | 7.5852          | 4.2419       | 5.7760     | 1.3838        | 3.1878             | 20.9966   | 0.7837          | 0.2761             | 2.7098             | 1.8715     | 3.2293                      | 10.8935      | 0.3514           | 0.5307      | 18.1424                 | 15.5101            | 9.5346              | 7.0668        |
| 0.7199 | 22000 | 5.7594        | 7.5562          | 4.1123       | 5.6151     | 1.3605        | 3.0954             | 21.0032   | 0.7640          | 0.2769             | 2.6019             | 1.8378     | 3.2377                      | 11.0744      | 0.3676           | 0.5431      | 18.2222                 | 15.7103            | 9.8826              | 7.2662        |
| 0.7526 | 23000 | 5.7118        | 7.4714          | 4.0531       | 5.5998     | 1.3546        | 3.0778             | 20.8820   | 0.7518          | 0.2800             | 2.7544             | 1.8756     | 3.2316                      | 10.9986      | 0.3571           | 0.5334      | 18.4476                 | 15.7161            | 9.6617              | 7.3730        |
| 0.7853 | 24000 | 5.8024        | 7.4414          | 4.0829       | 5.6335     | 1.3383        | 3.0710             | 20.8217   | 0.7487          | 0.2713             | 2.6091             | 1.8695     | 3.2365                      | 10.9929      | 0.3419           | 0.5213      | 18.4064                 | 15.7831            | 9.7747              | 7.4290        |
| 0.8181 | 25000 | 5.8608        | 7.4348          | 4.0571       | 5.5651     | 1.3294        | 3.0518             | 20.6831   | 0.7393          | 0.2784             | 2.6330             | 1.8293     | 3.2197                      | 10.9416      | 0.3484           | 0.5213      | 18.6359                 | 15.8463            | 9.6883              | 7.4697        |
| 0.8508 | 26000 | 5.742         | 7.4188          | 3.9483       | 5.4911     | 1.3288        | 3.0402             | 20.7187   | 0.7376          | 0.2772             | 2.6812             | 1.8540     | 3.2415                      | 10.9619      | 0.3560           | 0.5323      | 18.6388                 | 15.7688            | 9.6707              | 7.3793        |
| 0.8835 | 27000 | 5.7429        | 7.3956          | 3.9016       | 5.4393     | 1.3277        | 3.0129             | 20.6748   | 0.7314          | 0.2820             | 2.6526             | 1.8798     | 3.1869                      | 10.8744      | 0.3435           | 0.5228      | 18.5191                 | 15.7264            | 9.5707              | 7.4266        |
| 0.9162 | 28000 | 5.7825        | 7.3748          | 3.9100       | 5.4261     | 1.3420        | 3.0142             | 20.6013   | 0.7263          | 0.2764             | 2.6708             | 1.8529     | 3.1748                      | 10.8951      | 0.3491           | 0.5257      | 18.4914                 | 15.5663            | 9.6552              | 7.2807        |
| 0.9490 | 29000 | 5.5179        | 7.3555          | 3.9046       | 5.3902     | 1.3283        | 2.9882             | 20.5828   | 0.7169          | 0.2732             | 2.6742             | 1.8457     | 3.1760                      | 10.9126      | 0.3494           | 0.5246      | 18.5619                 | 15.6746            | 9.6539              | 7.3694        |
| 0.9817 | 30000 | 5.4044        | 7.3390          | 3.8742       | 5.3713     | 1.3127        | 2.9796             | 20.5703   | 0.7120          | 0.2669             | 2.5612             | 1.8536     | 3.1602                      | 10.9068      | 0.3464           | 0.5229      | 18.5389                 | 15.6788            | 9.5690              | 7.4148        |
| 1.0000 | 30560 | -             | 7.3346          | 3.8728       | 5.3680     | 1.3066        | 2.9780             | 20.5635   | 0.7107          | 0.2672             | 2.5046             | 1.8514     | 3.1596                      | 10.9153      | 0.3467           | 0.5233      | 18.5525                 | 15.6815            | 9.5687              | 7.4302        |

### Environmental Impact
Carbon emissions were measured using [CodeCarbon](https://github.com/mlco2/codecarbon).
- **Energy Consumed**: 0.506 kWh
- **Carbon Emitted**: 0.197 kg of CO2
- **Hours Used**: 3.163 hours

### Training Hardware
- **On Cloud**: No
- **GPU Model**: 1 x NVIDIA GeForce RTX 3090
- **CPU Model**: 13th Gen Intel(R) Core(TM) i7-13700K
- **RAM Size**: 31.78 GB

### Framework Versions
- Python: 3.11.6
- Sentence Transformers: 3.3.0.dev0
- Transformers: 4.45.2
- PyTorch: 2.5.0+cu121
- Accelerate: 1.0.0
- Datasets: 2.20.0
- Tokenizers: 0.20.1-dev.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->