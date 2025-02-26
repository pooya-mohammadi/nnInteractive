import torch


def cleanse_checkpoint_for_release(checkpoint: str):
    a = torch.load(checkpoint, weights_only=False)
    del a['optimizer_state']
    del a['init_args']['dataset_json']
    a['trainer_name']='nnInteractiveTrainer_stub'
    torch.save(a, checkpoint)

if __name__ == '__main__':
    cleanse_checkpoint_for_release('/omics/groups/OE0441/E132-Projekte/Projects/'
                                   '2025_Isensee_Rokuss_Kraemer_nnInteractive/checkpoints'
                                   '/nnInteractiveTrainerV2_educatedGuess__nnUNetResEncUNetLPlans_noResampling__3d_fullres_ps192_bs24/'
                                   'fold_0/checkpoint_final.pth')
    cleanse_checkpoint_for_release('/omics/groups/OE0441/E132-Projekte/Projects/'
                                   '2025_Isensee_Rokuss_Kraemer_nnInteractive/checkpoints'
                                   '/old_code_nnInteractiveTrainerV2_2000ep__nnUNetResEncUNetLPlans_noResampling__3d_fullres_ps192_bs24/'
                                   'fold_0/checkpoint_final.pth')
    cleanse_checkpoint_for_release('/omics/groups/OE0441/E132-Projekte/Projects/'
                                   '2025_Isensee_Rokuss_Kraemer_nnInteractive/checkpoints'
                                   '/old_code_nnInteractiveTrainerV2_5000ep__nnUNetResEncUNetLPlans_noResampling__3d_fullres_ps192_bs24/'
                                   'fold_0/checkpoint_final.pth')
