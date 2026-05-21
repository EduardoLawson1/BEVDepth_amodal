from nuscenes.nuscenes import NuScenes
import json
import os

# 1. Configuração de caminhos
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuScenes', verbose=True)
res_path = 'outputs/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema/results_nusc.json'
output_dir = 'outputs/visualizacoes_por_sample'

# Cria o diretório se não existir
os.makedirs(output_dir, exist_ok=True)

with open(res_path, 'r') as f:
    predictions = json.load(f)

tokens = list(predictions['results'].keys())[:5]

for sample_token in tokens:
    print(f"Processando token: {sample_token}")
    
    # Define os nomes de saída usando o sample_token
    cam_out = os.path.join(output_dir, f'cam_front_{sample_token}.png')
    bev_out = os.path.join(output_dir, f'bev_map_{sample_token}.png')
    
    # 3. Renderiza e salva
    # Renderiza a câmera frontal com as predições
    nusc.render_sample_data(nusc.get('sample', sample_token)['data']['CAM_FRONT'], 
                           out_path=cam_out)
    
    # Renderiza o mapa completo (BEV)
    nusc.render_sample(sample_token, out_path=bev_out)

print(f"\nSucesso! As imagens foram salvas em: {output_dir}")