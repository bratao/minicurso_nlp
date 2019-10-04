import kashgari
import re

from kashgari.embeddings import (
    NumericFeaturesEmbedding,
    BareEmbedding,
    StackedEmbedding,
)


def preprocess(word):
    # No paper original ele troca todos os numeros por 0
    word = re.sub('\d', '0', word)
    return word

def read_trained_lener():
    tokens = []
    tags = []
    MIN_LENGHT = 200
    with open("LeNer-Dataset/train.txt", 'r', encoding='utf-8') as f:
        actual_tokens = []
        actual_tags = []
        for line in f:
            if line == '\n':
                if len(actual_tags) > MIN_LENGHT:
                    tokens.append(actual_tokens)
                    tags.append(actual_tags)
                    actual_tokens = []
                    actual_tags = []
                continue
            token, tag = line.split(' ')
            actual_tokens.append(preprocess(token))
            actual_tags.append(tag.strip())

        if len(actual_tags) > 0:
            tokens.append(actual_tokens)
            tags.append(actual_tags)

    return tokens, tags


if __name__ == "__main__":

    tokens, labels = read_trained_lener()

    # TODO
    # 1 - Usar um Word Embedding, olha a classe kashgari.embeddings.WordEmbedding
    # Eu usaria o https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
    # ou o GloVE-300 do http://nilc.icmc.usp.br/embeddings se não der certo

    # 2 - Ver como fazer o Predict. Temos que processar a frase para ficar igual a deles.
    # Eles usam um PunktSentenceTokenizer com um abbrev_list. Esses scripts estao na pasta leNer-dataset.

    # 3 - Ver como integrar esse codigo com o webstruct atual
    # 4 - Seria uma boa ideia ter uma interface tipo o Broka. Para que existesse a lista de arquivos, e que
    # pudesse abrir para re-treinar, abrindo com o plugin de Ramon.
    # Uma ideia seria ate converter o dataset deles atual para o formato do broka hoje em Html ( pode ser algo simples, como colocar cada paragrafo como um p)

    # 5 - Fazer a persistencia ( O kashgari tem um metodo save/load)


    # 2 - Aumentar epochs para treinar

    # You can use WordEmbedding or BERTEmbedding for your text embedding
    text_embedding = BareEmbedding(task=kashgari.LABELING)

    text_embedding.analyze_corpus(tokens, labels)

    # Now we can embed with this stacked embedding layer
    # We can build any labeling model with this embedding

    from kashgari.tasks.labeling import BiLSTM_CRF_Model

    model = BiLSTM_CRF_Model(embedding=text_embedding)
    model.fit(tokens, labels, batch_size=8, epochs=10)

    print(model.predict(tokens))
    # print(model.predict_entities(x))
