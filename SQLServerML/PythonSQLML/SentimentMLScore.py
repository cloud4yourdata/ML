import numpy
import pandas
from microsoftml import rx_logistic_regression, rx_featurize, rx_predict, get_sentiment
# Create the data
customer_reviews = pandas.DataFrame(data=dict(review=[
            ":)"]))

## Get the sentiment scores
sentiment_scores = rx_featurize(
    data=customer_reviews,
    ml_transforms=[get_sentiment(cols=dict(scores="review"))])

# Let's translate the score to something more meaningful
sentiment_scores["eval"] = sentiment_scores.scores.apply(
            lambda score: "AWESOMENESS" if score > 0.6 else "BLAH")
print(sentiment_scores)
