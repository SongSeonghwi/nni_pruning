import torch



from nni.compression.pruning import L1NormPruner,LevelPruner
from nni.compression.speedup import ModelSpeedup


model1 = torch.load('./models/model.pth')
config_list1 = [{'sparse_ratio':0.5,
                 'op_types': ['Linear','Conv2d']
 }]
pruner1 = LevelPruner(model1, config_list1)
pruner1.compress()
pruner1.unwrap_model()

torch.save(model1, './models/level_pruned_model.pth')


model2 = torch.load('./models/model.pth') 
config_list2 = [{
    'op_types': ['Linear', 'Conv2d'],
    'exclude_op_names': ['fc3'],
    'sparse_ratio': 0.5
}]

pruner2 = L1NormPruner(model2, config_list2)

_, masks = pruner2.compress()
pruner2.unwrap_model()
ModelSpeedup(model2, torch.rand(64,1,28,28), masks).speedup_model()



torch.save(model2,'./models/L1_model.pth')
