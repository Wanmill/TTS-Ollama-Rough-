import torch
import json
import os

def extract_config_from_pth(pth_path, output_json="config_extracted.json"):
    ckpt = torch.load(pth_path, map_location="cpu")

    if "weight" not in ckpt:
        print("❌ File .pth tidak valid atau tidak mengandung 'weight'")
        return

    weight = ckpt["weight"]

    def get_shape(key):
        if key in weight:
            return list(weight[key].shape)
        return None

    print(f"🔍 Inspecting: {pth_path}\n{'='*50}")

    # Ambil shape dari layer penting
    shape_phone_emb = get_shape("enc_p.emb_phone.weight")
    shape_pitch_emb = get_shape("enc_p.emb_pitch.weight")
    shape_mel_decoder = get_shape("dec.noise_convs.0.weight")
    shape_speaker_emb = get_shape("emb_g.weight")

    if not shape_phone_emb or not shape_pitch_emb or not shape_mel_decoder:
        print("❌ Tidak ditemukan layer penting. Model mungkin corrupt.")
        return

    # Estimasi parameter dari weight
    n_mel_channels = shape_mel_decoder[2]
    spk_embed_dim = shape_speaker_emb[1] if shape_speaker_emb else 256
    n_speakers = shape_speaker_emb[0] if shape_speaker_emb else 0
    inter_channels = shape_pitch_emb[1]
    filter_channels = shape_pitch_emb[0]

    # Buat config dict
    config = {
        "train": {
            "sampling_rate": ckpt.get("sr", 48000),
            "segment_size": 8192
        },
        "data": {
            "filter_length": 1024,
            "hop_length": 300,
            "n_mel_channels": n_mel_channels
        },
        "model": {
            "type": "vits",
            "block_nums": 9,
            "resblock": "1",
            "n_layers": 4,
            "hidden_channels": inter_channels,
            "spk_embed_dim": spk_embed_dim,
            "gin_channels": 256,
            "inter_channels": inter_channels,
            "filter_channels": filter_channels,
            "n_heads": 2,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5]] * 3,
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "sr": ckpt.get("sr", 48000),
            "is_half": False,
            "use_spectral_norm": False,
            "n_speakers": n_speakers
        }
    }

    # Simpan ke file
    with open(output_json, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Config berhasil disimpan ke {output_json}")
    print(f"📊 n_mel_channels: {n_mel_channels}, filter_channels: {filter_channels}, n_speakers: {n_speakers}")

# Contoh pemakaian:
pth_path = "D:/VisualSC/MyChatbotRAG/rvc_models/Yuki/model.pth"
extract_config_from_pth(pth_path)
