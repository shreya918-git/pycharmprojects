import nlpcloud
class API:
    def ner(self,a,):

        client = nlpcloud.Client("finetuned-llama-3-70b", "81de71900e75318ad4795fa55d4f9214282707c7", gpu=True)
        response=client.summarization(text=a,
            size="small"
        )
        return response
    def sentiment(self,a):

        client = nlpcloud.Client("finetuned-llama-3-70b", "81de71900e75318ad4795fa55d4f9214282707c7", gpu=True)
        response=client.sentiment(
            text=a,
            target="NLP Cloud"
        )
        return response

    def headline(self,a):

        client = nlpcloud.Client("t5-base-en-generate-headline", "81de71900e75318ad4795fa55d4f9214282707c7", gpu=False)
        response=client.summarization(
            text=a
        )
        return response