import os
import time
import pdb

script_path = os.path.abspath(__file__)
proj_path = os.path.dirname(os.path.dirname(script_path))

################################ CIFAR100 ################################
### We currently only support single GPU usage. You can use this to specify which GPU you want to use
dev_id=0
### p_v = victim model dataset
p_v="CIFAR100"
### f_v = architecture of victim model
f_v="vgg16_bn"
### queryset = p_a = image pool of the attacker 
queryset="TinyImageNet200" #"TinyImageNet200"
### oeset = X_{OE} = outlier exposure dataset (Indoor67, SVHN)
oeset="SVHN"
### lambda of OE
oe_lamb=1.0
### Path to victim model's directory (the one downloded earlier)
vic_dir=f"models/victim/{p_v}-{f_v}-train-nodefense"
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=50000
### Batch size of queries to process for the attacker
ori_batch_size=32
lr=0.1
lr_step=10
lr_gamma=0.5
epochs=30
training_batch_size=32
### pretrained model
pretrained="imagenet"

begin_time = time.time()
# generate target model if not exist
# if not (os.path.exists(os.path.join(proj_path,vic_dir,'checkpoint.pth.tar')) 
#         and os.path.exists(os.path.join(proj_path,vic_dir,'model_poison.pt'))):
# status = os.system(f"python defenses/victim/train_admis.py {p_v} {f_v} -o {vic_dir} -b 64 -d {dev_id} -e 20 -w 4 --lr 0.01 --lr_step 10 --lr_gamma 0.5 --pretrained {pretrained} --oe_lamb {oe_lamb} -doe {oeset}")
#     if status != 0:
#         raise RuntimeError("Fail to generate target model!")

# Train composite model
# status = os.system(f"python defenses/victim/train_composite.py {p_v} {f_v} -o {vic_dir} -b 64 -d {dev_id} -e 20 -w 4 --lr 0.01 --lr_step 10 --lr_gamma 0.5 --pretrained {pretrained} --oe_lamb {oe_lamb} -doe {oeset}")

# Train EWE model
# status = os.system(f"python defenses/victim/train_ewe.py {p_v} {f_v} -o {vic_dir} -b 64 -d {dev_id} -e 20 -w 4 --lr 0.01 --lr_step 10 --lr_gamma 0.5 --pretrained {pretrained} --oe_lamb {oe_lamb} -doe {oeset}")

# Train MEA model
# status = os.system(f"python defenses/victim/train_mea.py {p_v} {f_v} -o {vic_dir} -b 64 -d {dev_id} -e 20 -w 4 --lr 0.01 --lr_step 10 --lr_gamma 0.5 --pretrained {pretrained} --oe_lamb {oe_lamb} -doe {oeset}")

end_time = time.time()

print('Training time = ', (end_time-begin_time)/20, flush=True)


for beta in [3.0]:

    query_list =  ['random']#['jbtr3','random']
    attack_list = ['naive']#['naive','s4l','smoothing','ddae','bayes','top1']#['naive','top1','s4l','smoothing','ddae','ddae+','bayes']
    defense_list = ['honeytunnel']#['none','rs','mad','am','top1','rounding','modelguard_w','modelguard_s']

    for policy in query_list:
        if policy == 'jbtr3':
            seedsize=10000
            jb_epsilon=0.1
            T=8
        for attack in attack_list:
            ### attack policy
            if policy == 'jbtr3':
                if attack in ['s4l','smoothing']:
                    continue
            if attack in ['ddae','ddae+','bayes']:
                defense_aware=1
            else:
                defense_aware=0
            
            if attack == 'top1':
                hardlabel=1
            else:
                hardlabel=0
            
            ## recovery setting
            if attack == 'ddae':
                shadow_model="alexnet"
                num_shadows=20
                shadowset="TinyImageNet200"
                num_classes=256 #100
                shadow_path=f"models/victim/{p_v}-{shadow_model}-shadow"
                recover_table_size=1000000
                recover_proc=5
                recover_params=f"'table_size:{recover_table_size};shadow_path:{shadow_path};recover_proc:{recover_proc};recover_nn:1'"
                generate_shadow=False
                if not os.path.exists(os.path.join(proj_path,shadow_path)):
                    generate_shadow = True
                else:
                    count = 0
                    for root, dirs, files in os.walk(os.path.join(proj_path,shadow_path)):
                        for file in files:
                            if file=='checkpoint.pth.tar':
                                count += 1
                    if count<num_shadows:
                        generate_shadow = True
                if generate_shadow:
                    # generate shadow models
                    status = os.system(f"python defenses/adversary/train_shadow.py {shadowset} {shadow_model} -o {shadow_path} -b 64 -d {dev_id} -e 5 -w 4 --lr 0.01 --lr_step 3 --lr_gamma 0.5 --pretrained {pretrained} --num_shadows {num_shadows} --num_classes {num_classes}")
                    if status != 0:
                        raise RuntimeError("Fail to generate shadow models for D-DAE!")
            elif attack == 'ddae+':
                recover_table_size=1000000
                concentration_factor=8.0
                recover_proc=5
                recover_params=f"'table_size:{recover_table_size};concentration_factor:{concentration_factor};recover_proc:{recover_proc};recover_nn:1'"
            else:
                recover_table_size=1000000
                recover_norm=1
                recover_tolerance=0.01
                concentration_factor=8.0
                recover_proc=5
                recover_params=f"'table_size:{recover_table_size};recover_norm:{recover_norm};tolerance:{recover_tolerance};concentration_factor:{concentration_factor};recover_proc:{recover_proc}'"

            ## semisupervised setting
            if attack == 's4l':
                semi_train_weight=1.0
            else:
                semi_train_weight=0.0
            semi_dataset=queryset

            ## augmentation setting
            if attack == 'smoothing':
                transform=1
                qpi=3
            else:
                transform=0
                qpi=1

            policy_suffix=f"_{attack}"

            for defense in defense_list:
                batch_size=ori_batch_size
                ### Defense strategy
                ## Quantization settings
                if defense == 'modelguard_s':
                    quantize=1
                    quantize_epsilon=1.0
                    
                else:
                    quantize=0
                    quantize_epsilon=0.0
                optim=0
                ydist="l1"
                frozen=0
                quantize_args=f"'epsilon:{quantize_epsilon};ydist:{ydist};optim:{optim};frozen:{frozen};ordered_quantization:1'"
                
                ## Perturbation settings
                if defense == 'none':
                    ## None
                    strat="none"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/none"
                    # Parameters to defense strategy, provided as a key:value pair string. 
                    defense_args=f"'out_path:{out_dir}'"

                elif defense == 'honeytunnel':
                    ## HoneyTunnel
                    pos=10
                    size=5
                    target_class=9
                    hard_label=1
                    strat="honeytunnel"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/honeytunnel"
                    # Parameters to defense strategy, provided as a key:value pair string. 
                    defense_args=f"'pos:{pos};size:{size};target_class:{target_class};hard_label:{hard_label};out_path:{out_dir}'"      
                        
                elif defense == 'rs':
                    ## reverse sigmoid
                    strat="reverse_sigmoid"
                    beta=0.02
                    gamma=0.2
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/revsig/beta{beta}-gamma{gamma}"
                    defense_args=f"'beta:{beta};gamma:{gamma};out_path:{out_dir}'"

                elif defense == 'dawn':
                    ## DAWN
                    strat="dawn"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/dawn"
                    defense_args=f"'out_path:{out_dir}'"
                    
                elif defense == 'composite':
                    ## COMPOSITE
                    strat="composite"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/composite"
                    defense_args=f"'out_path:{out_dir}'"

                elif defense == 'ewe':
                    ## EWE
                    strat="ewe"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/ewe"
                    defense_args=f"'out_path:{out_dir}'"

                elif defense == 'mea':
                    ## MEA
                    strat="mea"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/ewe"
                    defense_args=f"'out_path:{out_dir}'"    
                
                elif defense == 'mad':
                    strat="mad"
                    ydist="l1"
                    # Perturbation norm
                    eps=1.0
                    batch_size=1
                    # Perturbation mode: extreme|argmax
                    oracle="argmax"
                    # Initialization to the defender's surrogate model. 'scratch' refers to random initialization.
                    proxystate="scratch"
                    # Path to surrogate model
                    proxydir=f"models/victim/{p_v}-{f_v}-train-nodefense-{proxystate}-advproxy"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/mad_{oracle}_{ydist}/eps{eps}-proxy_{proxystate}"
                    # Parameters to defense strategy, provided as a key:value pair string. 
                    defense_args=f"'epsilon:{eps};batch_constraint:0;objmax:1;oracle:{oracle};ydist:{ydist};model_adv_proxy:{proxydir};out_path:{out_dir}'"
                    if not os.path.exists(os.path.join(proj_path,proxydir,"checkpoint.pth.tar")):
                        # generate proxy model
                        status = os.system(f"python defenses/victim/train.py {p_v} {f_v} --out_path {proxydir} --device_id {dev_id} --epochs 1 --train_subset 10 --lr 0.0")
                        if status != 0:
                            raise RuntimeError("Fail to generate proxy model for MAD!")
                
                elif defense == 'am':
                    strat="am"
                    defense_lv=0.4
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/am/tau{defense_lv}"
                    defense_args=f"'defense_level:{defense_lv};out_path:{out_dir}'"
                
                elif defense == 'top1':
                    ## topk
                    strat="topk"
                    topk=1
                    rounding=0
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/top{topk}"
                    defense_args=f"'topk:{topk};rounding:{rounding};out_path:{out_dir}'"

                elif defense == 'rounding':
                    strat="rounding"
                    rounding=1
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/rounding{rounding}"
                    defense_args=f"'rounding:{rounding};out_path:{out_dir}'"

                elif defense == 'modelguard_w':
                    ## MLD
                    strat="mld"
                    # Using Batch Constraint
                    batch_constraint=0
                    # Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
                    ydist="l1"
                    # Perturbation norm
                    eps=1.0
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/modelguardw/eps{eps}"
                    # Parameters to defense strategy, provided as a key:value pair string. 
                    defense_args=f"'epsilon:{eps};batch_constraint:{batch_constraint};ydist:{ydist};out_path:{out_dir}'"
                
                elif defense == 'modelguard_s':
                    strat="none"
                    # Output path to attacker's model
                    out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/modelguards/eps{quantize_epsilon}"
                    # Parameters to defense strategy, provided as a key:value pair string. 
                    defense_args=f"'out_path:{out_dir}'"
                
                # skip some pairs
                if defense == 'none' and defense_aware==1:
                    continue
                if policy == 'jbtr3' and defense in ['s4l','smoothing']:
                    continue
                
                command_eval = f"python defenses/victim/eval.py {vic_dir} {strat} {defense_args} --quantize {quantize} --quantize_args {quantize_args} --out_dir {out_dir} --batch_size {batch_size} -d {dev_id}"

                start_time = time.time()

                status = os.system(command_eval)

                end_time = time.time()

                print('Test time = ', (end_time-start_time), flush=True)

                if status != 0:
                    raise RuntimeError("Fail to evaluate the protected accuracy for defense {}".format(defense))
                if policy == 'random':
                    # (adversary) generate transfer dataset (only when policy=random)
                    command_transfer = f"python defenses/adversary/transfer.py {policy} {vic_dir} {strat} {defense_args} --out_dir {out_dir} --batch_size {batch_size} -d {dev_id} --queryset {queryset} --budget {budget} --quantize {quantize} --quantize_args {quantize_args} --defense_aware {defense_aware} --recover_args {recover_params} --hardlabel {hardlabel} --train_transform {transform} --qpi {qpi}"
                                            
                    # (adversary) train kickoffnet and evaluate
                    command_train = f"python defenses/adversary/train.py {out_dir} {f_v} {p_v} --budgets {budget} -e {epochs} -b {training_batch_size} --lr {lr} --lr_step {lr_step} --lr_gamma {lr_gamma} -d {dev_id} -w 4 --pretrained {pretrained} --vic_dir {vic_dir} --semitrainweight {semi_train_weight} --semidataset {semi_dataset}"

                    status = os.system(command_transfer)
                    if status != 0:
                        if not os.path.exists(os.path.join(out_dir,'params_transfer.json')):
                            raise RuntimeError("Fail to generate transfer set with attack {} and defense {}".format('random_'+attack,defense))
                    status = os.system(command_train)
                    if status != 0:
                        raise RuntimeError("Fail to train the substitute model with attack {} and defense {}".format('random_'+attack,defense))
                elif policy == 'jbtr3':
                    # (adversary) Use jbda-tr as attack policy
                    command_train = f"python defenses/adversary/jacobian.py {policy} {vic_dir} {strat} {defense_args} --quantize {quantize} --quantize_args {quantize_args} --defense_aware {defense_aware} --recover_args {recover_params} --hardlabel {hardlabel} --model_adv {f_v} --pretrained {pretrained} --out_dir {out_dir} --testdataset {p_v} -d {dev_id} --queryset {queryset} --query_batch_size {batch_size} --budget {budget} -e {epochs} -b {training_batch_size} --lr {lr} --lr_step {lr_step} --lr_gamma {lr_gamma} --seedsize {seedsize} --epsilon {jb_epsilon} --T {T}"
                    status = os.system(command_train)
                    if status != 0:
                        if not os.path.exists(os.path.join(out_dir,'params_transfer.json')):
                            raise RuntimeError("Fail to train the substitute model with attack {} and defense {}".format('jbtr3_'+attack,defense))
