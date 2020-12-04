## Welcome to My Page

![MyImg](images/mypic.png)

I have completed my B.Sc. from the Department of EEE, Bangladesh University of Engineering and Technology, in April 2019. Currently, I am working as a Machine Larning Engineer at [REVE Systems](https://www.revesoft.com/), Bangladesh. I am working in REVE Systems for almost a year now. Before joining REVE Systems, I was with the Digital Signal Processing Research Laboratory at the Department of EEE, Bangladesh University of Engineering and Technology.

I have done several projects, competitions and researches related to Machine Learning, Computer Vision, Biomedical Image Processing, Deep Learning, Audio Processing and Natural Language Processing. 

### Undergraduate Projects

My Passion for application based Machine Learning, Signal Processing, Image Processing research and development started from my third year of B.Sc. Education. I will explain my undergraduate projects one by one briefly. I will also give link to the codes for some projects.


#### Forensic Image Generation and Plotting Using CNC Plotter

In my fourth year, in Control Systems I Laboratory, I did a group project for generating images of crime suspects from facial attributes and plotting using CNC plotter. The idea was to do the job of a sketch artist in a forensic department. We used generative model to generate image from facial attributes and after processing the image the image was to be plotted in a 2D paper using CNC plotter. We designed and trained the generative model using [Celeb-A dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and built the CNC Plotter ourself with a limited budget.

Celeb-A dataset has over 200k+ dataset of celebrities with their facial attributes. There were a total of fourty facial attributes. We converted these facial attributes into a 40-dimensional binary vector. We designed an encoder to transform this encoded vector into a latent representation. We also designed a ResNet based Generator Network to transform this latent representation into an image. The generative model was trained using Celeb-A dataset. Our network was trained using a MSE-Loss coupled with Adversarial-Loss. Some sample output of validation set is given below.

![Image1](images/sample-2.png) ![Image2](images/sample-3.png)

The top row is the reference images and the bottom row is the generated images with the facial attributes of refernce images given as input. The generated images are not the sharpest and clearest images in quality, but was sufficient to impress our teacher and peers. These are the generated images for the validation set images, and you must be wondering what is the performance with random facial attributes given as input. The code for this project can be found in this [link](https://github.com/sabbiracoustic1006/Generation-of-Face-Images-from-Facial-Attributes). Lets run the code with a random but carefully chosen input facial attributes. 


```markdown
# The attributes list is given below
attribute_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',' Bags_Under_Eyes', 'Bald', 'Bangs', 
                  'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                  'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                  'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                  'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
                  'Wearing_Necktie', 'Young']


# Step for generating an image for an attractive Female with brown hair and heavy makeup
python inference.py --attributes 'brown_hair heavy_makeup attractive no_beard' \
                    --encoder saved_models/vae.pth --generator saved_models/generator.pth \
                    --device cpu 
                    
# it is a bit non sense that no_beard has to be given as input for generating image of a female,
# the attributes processing has to be improved :)
# A generated image will be saved in generated-imgs folder with the same name as the attributes
```
![Image3](images/brown_hair_heavy_makeup_attractive_no_beard.jpg)   

Now let's see what happens when we add young attribute. The code is given below.

```markdown
python inference.py --attributes 'brown_hair heavy_makeup attractive no_beard young' \
                    --encoder saved_models/vae.pth --generator saved_models/generator.pth \
                    --device cpu 
                    
# A younger female's image is generated hopefully
```
![Image4](images/brown_hair_heavy_makeup_attractive_no_beard_young.jpg)

Now let's see what happens if we add eyeglasses attribute. The code is given below.

```markdown
python inference.py --attributes 'brown_hair heavy_makeup attractive no_beard young eyeglasses' \
                    --encoder saved_models/vae.pth --generator saved_models/generator.pth \
                    --device cpu 
                    
```
![Image5](images/brown_hair_heavy_makeup_attractive_no_beard_young_eyeglasses.jpg)
 
Indeed she is wearing glasses now. Now let's see how does she looks while smiling. The code is given below.

```markdown
python inference.py --attributes 'brown_hair heavy_makeup attractive no_beard young eyeglasses smiling' \
                    --encoder saved_models/vae.pth --generator saved_models/generator.pth \
                    --device cpu 
                    
```
![Image6](images/brown_hair_heavy_makeup_attractive_no_beard_young_eyeglasses_smiling.jpg)

Finally the smile :) ... The model shows generalized performance.
 
A Video of CNC Plotter is given below. Click on the image to see the video.

[![CNC PLOTTER](http://i3.ytimg.com/vi/dK8GlewlfwY/maxresdefault.jpg)](https://www.youtube.com/watch?v=dK8GlewlfwY)

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/sabbiracoustic1006/sabbiracoustic1006.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
