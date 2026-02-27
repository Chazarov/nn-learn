from models.weights import WeightsData, WeightsMeta

class WeightsRepository:

    def create(self, user_id: str, weights: WeightsData) -> WeightsMeta:
        ...
    
    def delete(self, id:str):
        ...