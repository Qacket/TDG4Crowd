from importlib import import_module


def generating_fixed_task(crowd_file, truth_file, exist_task, generate_method_name, generate_file, train_loader):
    mod = import_module(generate_method_name + '.method')
    em = getattr(mod, generate_method_name)(crowd_file, truth_file)
    em.run()
    em.generate_fixed_task(exist_task, generate_file, train_loader)

def generating_fixed_annotator(crowd_file, truth_file, exist_annotator, generate_method_name, generate_file, train_loader):
    mod = import_module(generate_method_name + '.method')
    em = getattr(mod, generate_method_name)(crowd_file, truth_file)
    em.run()
    em.generate_fixed_annotator(exist_annotator, generate_file, train_loader)


def generating(crowd_file, truth_file, sample_file, generate_method_name, generate_file, train_loader):

    mod = import_module(generate_method_name + '.method')
    em = getattr(mod, generate_method_name)(crowd_file, truth_file)
    em.run()
    em.generate(sample_file, generate_file, train_loader)


def generating_replenish(crowd_file, truth_file, exist_task, generate_method_name, generate_file, train_loader):
    mod = import_module(generate_method_name + '.method')
    em = getattr(mod, generate_method_name)(crowd_file, truth_file)
    em.run()
    em.generate_replenish(exist_task, generate_file, train_loader)


def generating_for_distribution(crowd_file, truth_file, generate_method_name, generate_file, generate_truth_file, number, workers_number):
    mod = import_module(generate_method_name + '.method')
    em = getattr(mod, generate_method_name)(crowd_file, truth_file)
    em.run()
    em.generate_for_distribution(generate_file, generate_truth_file, number, workers_number)
