from plantseg.models.zoo import model_zoo

def test_get_model_by_id():
    model_zoo.get_model_by_id('efficient-chipmunk')
