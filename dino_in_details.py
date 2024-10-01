import torch
import numpy as np
import cv2

vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

image = cv2.imread("img.png")
image_size = (224, 224)
blob = cv2.dnn.blobFromImage(image, 1.0/255, image_size, swapRB=True, crop=False)
blob[0][0] = (blob[0][0] - 0.485)/0.229
blob[0][1] = (blob[0][1] - 0.456)/0.224
blob[0][2] = (blob[0][2] - 0.406)/0.225

blob = torch.tensor(blob) # 1 x 3 x 224 x 224
x = blob

# Prepare tokens
# -------------------------------------------------
B, _, h, w = x.shape 
x = vits8.patch_embed(x)  # patch linear embedding by Conv2d(3, 384, kernel_size=(8, 8), stride=(8, 8))
print(x.shape) # 1 x 784 x 384   # 784 = 28 x 28, 224 = 28 x 8, 384 = 6 x 64

# example of the first patch embedding 
patch = blob[0][:,:8,:8].reshape(-1,1) # 3 x 8 x 8 -> 192, 1
weights = vits8.patch_embed.proj.weight.reshape(384,-1) # 384 x 3 x 8 x 8 -> 384 x 192
biases = vits8.patch_embed.proj.bias.reshape(384,1) # 384 -> 384 x 1
hidden_state = weights@patch + biases
hidden_state = hidden_state.reshape(-1) # 384 x 1 -> 384
(x[0][0]-hidden_state).abs().max() < 0.000001 # True

# back projection of the first hidden state
hidden_state = x[0][0].reshape(-1,1)
patch2 = torch.linalg.pinv(weights)@(hidden_state-biases)
(patch-patch2).abs().max() < 0.0001 # True

# add the [CLS] token to the embed patch tokens
cls_tokens = vits8.cls_token.expand(B, -1, -1)
x = torch.cat((cls_tokens, x), dim=1) # 1 x 785 x 384

# back projection of CLS
inverse_weights = torch.linalg.pinv(weights)
def backproj(hidden_state):
    patch = inverse_weights@(hidden_state-biases)
    blob = patch.reshape(3,8,8).detach().cpu().numpy()
    blob[0] = blob[0]*0.229 + 0.485
    blob[1] = blob[1]*0.224 + 0.456
    blob[2] = blob[2]*0.225 + 0.406
    blob = (blob*255).astype(np.uint8)
    image = cv2.merge([blob[2],blob[1],blob[0]])
    return image

cls_image = backproj(vits8.cls_token.reshape(-1,1))
cv2.imwrite('cls.png',cls_image)

# what is length of embeddings?
norms=torch.norm(x,dim=2)
print(norms.mean(), norms.std()) # 1.52, 0.29

# add positional encoding to each token
print(vits8.pos_embed.shape) # 1 x 785 x 384
# vits8.interpolate_pos_encoding(x, h, w) == vits8.pos_embed.shape for h, w = 224, 224
x = x + vits8.interpolate_pos_encoding(x, h, w) # == x + vits8.pos_embed # 1 x 785 x 384

x = vits8.pos_drop(x)

def display(x):
    disp = np.zeros((224,224,3),np.uint8)
    for r in range(28):
        for c in range(28):
            subimage = backproj(x[0][r*28+c+1].reshape(-1,1))
            disp[8*r:8*r+8,8*c:8*c+8,:] = subimage
    return disp

# Transformer encoder
# -------------------------------------------------
attn_maps = []
cls_images = []
for i, block in enumerate(vits8.blocks):
    cls_image = backproj((x - vits8.pos_embed)[0][0].reshape(-1,1))
    cv2.imwrite(f'cls{i}.png',cls_image)
    cls_images.append(cls_image)
    disp_image = display(x - vits8.pos_embed)
    cv2.imwrite(f'disp{i}.png',disp_image)
    x = block.norm1(x)
    y, attn = block.attn(x)
    attn_maps.append(attn)
    x = x + block.drop_path(y)
    x = x + block.drop_path(block.mlp(block.norm2(x)))

cls_image = backproj((x - vits8.pos_embed)[0][0].reshape(-1,1))
cv2.imwrite('cls-final.png',cls_image)
cls_images.append(cls_image)
disp_image = display(x - vits8.pos_embed)
cv2.imwrite('disp-final.png',disp_image)
x = vits8.norm(x) # 1 x 785 x 384

features = x[:, 0] # 1 x 384
 
# Present results
# -------------------------------------------------

# Present features
print(features.shape)
print(features)

# Present attention maps.
attn_mat = torch.stack(attn_maps).squeeze(1) # 12 x 6 x 785 x 785
grid_size = int(np.sqrt(attn_mat.size(-1)))
rows = []
for i, attn_heads in enumerate(attn_mat):
    cols = []
    for j, attn_head in enumerate(attn_heads):
        mask = attn_head[0, 1:].pow(0.4).reshape(grid_size,grid_size).detach().cpu().numpy() # 28 x 28
        mask = (mask*255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask,(0,0),(mask.shape[1],mask.shape[0]),(255,255,255),2)
        cv2.putText(mask,f'H{j}',(mask.shape[1]*4//10,20),0,0.8,(255,255,255))
        cols.append(mask)
    
    row = cv2.vconcat(cols)
    cv2.putText(row,f'{i}',(5,30),0,1.0,(255,255,255),2)
    rows.append(row)

disp = cv2.hconcat(rows)
cv2.imwrite(f'att.png',disp)

#--------------------

imgs = [
    cls_images[:7],
    [255*np.ones_like(cls_images[0])] + cls_images[7:]
]

rows = []
for row in imgs:
    cols = []
    for col in row:
        cols.append(cv2.resize(col,(10*col.shape[1],10*col.shape[0]),interpolation=cv2.INTER_NEAREST))
        cols.append(255*np.ones((10*col.shape[1],col.shape[0],3),np.uint8))
    rows.append(cv2.hconcat(cols))
    rows.append(255*np.ones_like(rows[-1][:8,:,:]))

fin = cv2.vconcat(rows)
cv2.imwrite('fin.png',fin)