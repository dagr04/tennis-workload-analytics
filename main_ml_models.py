from workload_generator import WorkloadGenerator  
from ml_models import compare_models              

def main():
    generator = WorkloadGenerator("Alex","01-01-2022", 1000)
    data = generator.run() 
    model = compare_models(data)
    return model

if __name__ == "__main__":
    main()