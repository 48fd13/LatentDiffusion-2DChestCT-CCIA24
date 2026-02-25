import glob
import numpy as np
import os
import pylidc as pl
from pylidc.utils import consensus
from PIL import Image
from tqdm import tqdm
import json

def array2string(np_array, precision=2):
    s = np.array2string(np_array, precision=precision, separator=" ")
    s = s.replace("[", "").replace("]", "")
    return s

def string2array(s, dtype=np.float16):
    np_array = np.fromstring(s, dtype=dtype, sep=" ")
    return np_array

def save_image(input_np_array, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = Image.fromarray(input_np_array)
    x.save(out_path)

def normalize_lidc_img(img):
    min_ = -1000 # = "air" in Hounsfield unit (HU) https://en.wikipedia.org/wiki/Hounsfield_scale
    max_ = 2500 # in the LDIC dataset the largest HU in a nodule is ~2400 (however values above 1500 are exceptional)
    image = (img - min_)/(max_ - min_)
    image = np.clip(image, 0, 1)
    #image = (img - img.min())/(img.max() - img.min())
    image = np.dstack([image, image, image])
    return image

def get_nodule_and_masks_from_patient(patient_id, out_dir, out_format="png", clevel=0.25, crop_size=64, display=False):
    # e.g. patient_id = 'LIDC-IDRI-0078'
    # clevel = 0.25 means if 1/4 doctors marked it then we keep that part
    # amount of padding in each direction
    
    # Query for a scan, and convert it to an array volume.
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    vol = scan.to_volume(verbose=False)
    
    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations(verbose=True)
    img_list, mask_list, nod_id_list, res_list, bbox_list, nod_area_list = [], [], [], [], [], []
    
    feature_labels = ["sphericity", "margin", "lobulation", "spiculation", "texture"]
    d = []

    cbbxs, cmasks = [], []
    mask_vol = np.zeros_like(vol)
    for nod_idx, anns in enumerate(nods):
        cmask, cbbox, _ = consensus(anns, clevel=clevel, crop_size=(crop_size, crop_size))
        mask_vol[cbbox[0].start:cbbox[0].stop, cbbox[1].start:cbbox[1].stop, cbbox[2].start:cbbox[2].stop] = cmask
        cbbxs.append(cbbox)
        cmasks.append(cmask)

    for nod_idx, (anns, cbbox, cmask) in enumerate(zip(nods, cbbxs, cmasks)):
        
        local_d = {}
        # Perform a consensus consolidation and 50% agreement level.
        # We pad the slices to add context for viewing.
        #cmask,cbbox,_ = consensus(anns, clevel=clevel, pad=[(pad,pad), (pad,pad), (0,0)])
        #cmask, cbbox, _ = consensus(anns, clevel=clevel, crop_size=(crop_size, crop_size))
        #cmask, cbbox, _ = consensus(anns, clevel=clevel, crop_size=None)
        
        # Get the central slice of the computed bounding box
        k = int(0.5*(cbbox[2].stop - cbbox[2].start))
        bbx = np.array([[cbbox[i].start, cbbox[i].stop] for i in range(3)]).ravel()
        
        nod_id = patient_id + "-{:03d}-{:03d}".format(nod_idx + 1, cbbox[2].start + k)
            
        mask_crop = cmask[:, :, k]
        img_crop = vol[cbbox][:,:,k]
        if (np.prod(img_crop.shape) != crop_size**2) or (np.prod(mask_crop.shape) != crop_size**2):
            print(f"warning: something weird happened for {nod_id}. ignoring it...")
            continue
        
        # Check there are at least 2 valid annotations per nodule feature
        all_ann_ok = True

        for l in feature_labels:
            vals = np.array([getattr(ann, l) for ann in anns])
            if len(vals) < 2 or len(vals) > 4:
                all_ann_ok = False
            preset_vals = np.zeros(10)
            preset_vals[:len(vals)] = vals
            local_d[l] = array2string(preset_vals[:4], precision=1)
 
        if not all_ann_ok:
            continue
            
        img = vol[:,:,cbbox[2].start + k]
        mask = mask_vol[:, :, cbbox[2].start + k]
        nod_area = np.sum(mask.ravel()) #* scan.pixel_spacing ** 2
        local_d["area"] = f"{nod_area}"
        
        # resize both CT slice and mask to fixed resolution
        target_res = 0.6 # in mmm
        zoom_factor = scan.pixel_spacing/target_res
        new_size = int(img.shape[0] * zoom_factor)
        mask = np.array(Image.fromarray(mask).resize((new_size, new_size), Image.NEAREST))
        img = np.array(Image.fromarray(img).resize((new_size, new_size), Image.LANCZOS))

        img_path = os.path.join(out_dir, f"nodules/{nod_id}.{out_format}")
        #img_path = os.path.abspath(img_path)
        local_d["file_name"] = f"{nod_id}.{out_format}"
        img = normalize_lidc_img(img)
        if out_format == "png":
            img = (img * 255).astype(np.uint8)
        save_image(img, img_path)
        mask_path = os.path.join(out_dir, f"masks/{nod_id}.{out_format}")
        #mask_path = os.path.abspath(mask_path)
        #local_d["mask"] = f"{nod_id}.{out_format}"
        if out_format == "png":
            mask = (mask * 255).astype(np.uint8)
        save_image(mask, mask_path)

        local_d["pixel_spacing"] = f"{scan.pixel_spacing:.5f}" #CT resolution in mm
        local_d["slice_spacing"] = f"{scan.slice_spacing:.5f}"
        local_d["bbx"] = array2string(bbx, precision=1)
        
        d.append(local_d)
       
    return d


def read_jsonl(metadata_path):
    # returns a list of dictionaries
    with open(metadata_path, 'r') as f: 
        metadata = list(f)
    metadata = [json.loads(s) for s in metadata]
    return metadata

def write_jsonl(metadata_path, metadata):
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

def update_jsonl(metadata, metadata_path):
    if os.path.exists(metadata_path):
        metadata_ = read_jsonl(metadata_path)
        metadata = metadata + metadata_
        unique_tuples = set(tuple(sorted(d.items())) for d in metadata)
        unique_dicts = [dict(t) for t in unique_tuples]
        metadata = sorted(unique_dicts, key=lambda x: x['file_name'])
        os.remove(metadata_path)
    write_jsonl(metadata_path, metadata)


def main():
    LIDC_data_path = "/media/share/Datasets/LIDC-IDRI/LIDC-IDRI"
    out_dir = "paper_dataset_all"

    patient_ids = [p.split("/")[-1] for p in glob.glob(os.path.join(LIDC_data_path, "*"))]
    patient_ids = [p for p in sorted(patient_ids) if "LIDC-IDRI-" in p]
    crop_size = 64
    clevel = 0.2

    metadata_path = os.path.join(out_dir, "nodules/metadata.jsonl")
    save_freq = 10

    metadata = []
    for i, patient_id in enumerate(tqdm(sorted(patient_ids))):
        d = get_nodule_and_masks_from_patient(patient_id, out_dir, clevel=clevel, crop_size=crop_size, display=False)
        for local_d in d:
            metadata.append(local_d)
        if i % save_freq == 0:
            # update metadata
            update_jsonl(metadata, metadata_path)
    update_jsonl(metadata, metadata_path)
    
if __name__ == "__main__":
    main()
