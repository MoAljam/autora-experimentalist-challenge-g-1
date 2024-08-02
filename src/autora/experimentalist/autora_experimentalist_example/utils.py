from scipy.special import rel_entr
def divergence_scorer(conditions: Union[pd.DataFrame, np.ndarray],
                             models: List,
                             num_samples: Optional[int] = None,
                             ) -> float:
    
    # condtion_pool = conditions.copy()
    model_new = models[-1]
    model_old = models[-2]
    score = mean(sum(rel_entr(model_new.predict(conditions), model_old.predict(conditions))))
    
    print("score:", score)

    return score