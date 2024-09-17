from core.datasets import dataset_registry, CSVDataset

_features = [
        'WarTrain',
        'EmptyTrain', 
        'PassengerTrain', 
        'LongFreightTrain', 
        'RuralTrain', 
        'MixedTrain', 
        'ReinforcedCar', 
        'PassengerCar', 
        'EmptyWagon', 
        'LongWagon', 
        'LongTrain', 
        'FreightTrain', 
        'FreightWagon', 
        'OpenRoofCar', 
        'HexagonCar', 
        'SuperellipseCar', 
        'TrapezoidCar', 
        'EllipseCar', 
        'BoxCar', 
        'SawRoof', 
        'TriangularRoof', 
        'FlatRoof', 
        'HexagonCargo', 
        'DiamondCargo', 
        'InvTriangleCargo', 
        'TriangleCargo', 
        'XLSquareCargo', 
        'MSquareCargo', 
        'SSquareCargo', 
        'TwoWheelsWagon', 
        'ThreeWheelsWagon', 
        'FourWheelsWagon', 
        'XSWheels', 
        'SWheels', 
        'MWheels', 
        'LWheels', 
        #'CouplerHeight', #TODO rethink numeric features
        #'WagonSpacing', 
        #'Angle', 
        #'TotalCargoQuantity', 
        #'NumberOfWagons', 
        #'NumberOfLongWagons', 
        #'NumberOfFreightWagons', 
        #'NumberOfPassengerCars'
]

def _except(*features):
    return [f for f in _features if f not in features]

dataset_registry['xtrains_concepts_test_1'] = CSVDataset(
    "data/xtrains_rn.csv",
    target=["TypeA"],
    random_state=42,
    features=_except()
)
dataset_registry['xtrains_concepts_test_2'] = CSVDataset(
    "data/xtrains_rn.csv",
    target=["TypeA"],
    random_state=42,
    features=_except('WarTrain', 'EmptyTrain')
)