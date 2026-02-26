from typing import Any, Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile

from activation import ActivationType, ACTIVATIONS
from mathh.mv import init_perceptrone

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/learn/")
def learn_perceptrone(
    file_id: str,
    hidden_layers_architecture: List[int],
    activation_type: ActivationType

):
    activation = ACTIVATIONS[activation_type]
    # количество нейронов входного и выходного слоя узнаем из файла csv.
    # формат csv:
    # столбцы: первый - id, далее - название столбца - название свойства, последний столбец - классы
    #  по n-2 количеству столбцов вычисляем число свойств, тоесть нейронов входного слоя.
    #  по числу классов , доступных в файле - количество выходных классов, тоесть нейронов выходного слоя 

    input_layer_size:int = ...
    output_layer_size:int = ...

    architecture:List[int] = [input_layer_size]
    for i in range(len(hidden_layers_architecture)):
        architecture.append(hidden_layers_architecture[i])

    perceptrone = init_perceptrone(architecture)

    # далее обучение перцептрона и созранение его в файл в директорию data/weights

    return {"perceptrone_id": perceptrone_id}


@router.get("get_answer")
def get_answer(
    perceptrone_id: str,
    input_vector: List[float],
    activation_type: ActivationType
):
    activation = ACTIVATIONS[activation_type]

    return {"output": output_vector}


    
@router.post("/upload/scv")
async def upload_csv(file: UploadFile = File(..., description="CSV a training sample")) -> Dict[str, Any]:
    # Проверяем расширение
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Только CSV файлы!")
    
    # Созраняем файл в директорию data/learn присваиваем имя - {id}.csv
    
    return {"file_id": file_id}


@router.get("/files")
async def get_all_samples()-> Dict[str, Any]:

    # получение всех файлов из директории data/learn

    return { 
        "files": [
            "id": name.replace(".csv", ""),
            "name": name]  for name in file_names
    }


@router.get("/weights")
async def get_all_weights()-> Dict[str, Any]:

    # получение всех файлов из директории data/weights

    return { 
        "files": [
            "id": name.replace(".csv", ""),
            "name": name]  for name in file_names
    }
