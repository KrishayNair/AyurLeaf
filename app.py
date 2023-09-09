import streamlit as st
import os
from pathlib import Path
from streamlit_option_menu import option_menu
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AyurLeafAI",page_icon=":herb:",layout="wide")
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('plant_model_final.h5')

# Create label mapping based on subdirectory names
main_data_dir = 'Segmented Medicinal Leaf Images\Segmented Medicinal Leaf Images'
label_mapping = {i: label for i, label in enumerate(sorted(os.listdir(main_data_dir)))}

selected = option_menu(
    menu_title=None,
    options=["Predict","About"],
    icons=["search","book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)
if selected == "Predict":
# Streamlit UI
    st.title("AyurLeafAI")

    # Upload an image for prediction
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Check if the uploaded file is an image
        if uploaded_image.type.startswith('image/'):
            # Display the uploaded image
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess and predict the image
            def preprocess_image(image):
                image = load_img(image, target_size=(224, 224))
                image_array = img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)
                preprocessed_image = preprocess_input(image_array)
                return preprocessed_image

            if st.button("Predict"):
                preprocessed_image = preprocess_image(uploaded_image)
                predictions = model.predict(preprocessed_image)

                # Map model's numeric predictions to labels
                predicted_label_index = np.argmax(predictions)
                predicted_label = label_mapping[predicted_label_index]

                # Calculate accuracy (confidence)
                confidence = np.max(predictions)  # The highest predicted probability

                # Check if the predicted label is among the valid plant labels
                valid_plant_labels = label_mapping.values()
                if predicted_label in valid_plant_labels:
                    # Display prediction and accuracy
                    # st.write(f"Predicted Label: {predicted_label}")
                    # st.write(f"Accuracy: {confidence * 100:.2f}%")
                    accuracy= f"{confidence * 100:.2f}"
                string = "Predicted Ayurvedic Leaf  :-" + " " + predicted_label 
                percentage = "Accuracy =" + " " + accuracy + " %"
                if uploaded_image is None:
                    st.text("please upload an image")
                else:
                    st.success(string)
                    if confidence >= 0.75:
                        st.success(percentage)
                    elif confidence < 0.75:
                        st.warning(percentage)
                        
                if predicted_label == "Azadirachta Indica (Neem)":
                    st.subheader("Medicinal Uses")
                    st.write("Neem leaf is used for leprosy, eye disorders, bloody nose, intestinal worms, stomach upset, loss of appetite, skin ulcers, diseases of the heart and blood vessels (cardiovascular disease), fever, diabetes, gum disease (gingivitis), and liver problems. The leaf is also used for birth control and to cause abortions.")
                elif predicted_label == "Alpinia Galanga (Rasna)":
                    st.subheader("Medicinal Uses")
                    st.write("it is used in a variety of Ayurvedic remedies to treat rheumatism and inflammatory disorders, as well as other illnesses such as dementia, diabetes, and neurological diseases")
                elif predicted_label == "Amaranthus Green":
                    st.subheader("Medicinal Uses")
                    st.write("They help prevent birth defects in newborn babies and are needed for optimal mental and physical health")
                elif predicted_label == "Amaranthus Red":
                    st.subheader("Medicinal Uses")
                    st.write("It is used for ulcers, diarrhea, swelling of the mouth or throat, and high cholesterol")
                elif predicted_label ==  "Amaranthus Viridis (Arive-Dantu)":
                    st.subheader("Medicinal Uses")
                    st.write("Amaranthus viridis is used as traditional medicine in the treatment of fever, pain, asthma, diabetes, dysentery, urinary disorders, liver disorders, eye disorders and venereal diseases. The plant also possesses anti-microbial properties")
                elif predicted_label ==  "Artocarpus Heterophyllus (Jackfruit)":
                    st.subheader("Medicinal Uses")
                    st.write("The several parts of jack tree including fruits, leaves, and barks have been extensively used in traditional medicine due to its anticarcinogenic, antimicrobial, antifungal, anti-inflammatory, wound healing, and hypoglycemic effects")
                elif predicted_label ==  "Basella Alba (Basale)":
                    st.subheader("Medicinal Uses")
                    st.write("Basella alba is reported to improve testosterone levels in males, thus boosting libido. Decoction of the leaves is recommended as a safe laxative in pregnant women and children. Externally, the mucilaginous leaf is crushed and applied in urticaria, burns and scalds.")
                elif predicted_label ==  "Brassica Juncea (Indian Mustard)":
                    st.subheader("Medicinal Uses")
                    st.write("It is a folk remedy for arthritis, foot ache, lumbago and rheumatism. In China the seed is used as medicine against tumours. Ingestion may impart a body odour repellent to mosquitoes. Leaves applied to the forehead are said to relieve headache.")
                elif predicted_label ==  "Carissa Carandas (Karanda)":
                    st.subheader("Medicinal Uses")
                    st.write("In ethnomedicine, different parts of C. carandas have been used to treat anorexia, asthma, brain disease, constipation, cough, diarrhea, epilepsy, fever, leprosy, malaria, myopathic spams, pain, pharyngitis, scabies, and seizures")
                elif predicted_label ==  "Balloon vine":
                    st.subheader("Medicinal Uses")
                    st.write(" It is used for treating itching scalp and dandruff and hair growth")
                elif predicted_label ==  "Black Night Shade":
                    st.subheader("Medicinal Uses")
                    st.write("black nightshade has been used for stomach irritation, cramps, spasms, pain, and nervousness.")
                elif predicted_label ==  "Citrus Limon (Lemon)":
                    st.subheader("Medicinal Uses")
                    st.write(" known from traditional medicine, include treatment of high blood pressure, the common cold, and irregular menstruation. Moreover, the essential oil of C. limon is a known remedy for coughs")
                elif predicted_label ==  "Coriander Leaves":
                    st.subheader("Medicinal Uses")
                    st.write(" It is used for relieving gastrointestinal discomfort, respiratory, and urinary complaints")
                elif predicted_label ==  "Country Mallow":
                    st.subheader("Medicinal Uses")
                    st.write("a soothing agent that counters irritation and mild inflammation. Both mallow leaf and flower preparations are approved by the German Commission E for relief of sore throats and dry coughs. Mallow is typically used as a tea or gargle for these indications")
                elif predicted_label ==  "Crown flower":
                    st.subheader("Medicinal Uses")
                    st.write("The plant has been used traditionally to treat various ailments such as fever, cough, cold, asthma, bronchitis, etc. Ekka is a natural remedy for cough and cold. It contains anti-inflammatory properties and is effective against respiratory tract infections. It helps reduce inflammation and swelling in the lungs")
                elif predicted_label ==  "Dwarf Copperleaf (Green)":
                    st.subheader("Medicinal Uses")
                    st.write("Dwarf Copperleaf (Acalypha reptans), also known as Green Copperleaf, is occasionally used in traditional medicine. Its leaves may be applied topically for wound healing, minor skin inflammations, and pain relief. Some cultures have used it in herbal remedies for respiratory issues and as a potential anti-diabetic agent, although scientific evidence is limited. Always consult a healthcare professional before using it for medicinal purposes, as its effectiveness and safety can vary.")
                elif predicted_label ==  "Dwarf copperleaf (Red)":
                    st.subheader("Medicinal Uses")
                    st.write("Dwarf Copperleaf (Acalypha pendula), commonly known as Red Copperleaf, is primarily an ornamental plant, but it has limited traditional medicinal uses. Its leaves have been applied topically for treating minor skin irritations and wounds in some cultures. However, scientific evidence supporting its medicinal efficacy is scarce. Caution should be exercised when considering its use for medicinal purposes, and consultation with a healthcare professional is advisable")
                elif predicted_label ==  "Ficus Auriculata (Roxburgh fig)":
                    st.subheader("Medicinal Uses")
                    st.write("Ficus auriculata, or Roxburgh fig, has been used traditionally in various cultures for its potential medicinal benefits. It is believed to possess anti-inflammatory properties and has been used for wound healing, digestive issues, respiratory ailments, and as a potential remedy for diabetes and parasitic infections. However, scientific evidence supporting these uses is limited, and caution should be exercised. Consultation with a healthcare professional is advisable before considering it for medicinal purposes.")
                elif predicted_label ==  "Ficus Religiosa (Peepal Tree)":
                    st.subheader("Medicinal Uses")
                    st.write("Ficus religiosa, or the Peepal tree, is revered in several traditional medicinal systems. Its bark and leaves are believed to have anti-inflammatory properties and are used to treat various ailments like wounds, digestive disorders, respiratory problems, and diabetes in folk medicine. However, scientific evidence supporting these uses is limited, and consulting a healthcare professional is essential before using Peepal tree remedies for medicinal purposes due to potential side effects.")
                elif predicted_label ==  "Giant Pigweed":
                    st.subheader("Medicinal Uses")
                    st.write("Giant Pigweed (Amaranthus giganteus) has limited documented medicinal uses. Some traditional practices suggest using its leaves for poultices to alleviate skin irritations and insect bites. However, scientific evidence supporting its medicinal efficacy is scarce. It's crucial to exercise caution and consult with a healthcare professional before using Giant Pigweed for medicinal purposes, as it may have potential adverse effects or allergies in some individuals.")
                elif predicted_label ==  "Gongura":
                    st.subheader("Medicinal Uses")
                    st.write("Gongura, also known as Roselle or Hibiscus sabdariffa, has been used traditionally in some regions for its potential medicinal properties. Its leaves are rich in antioxidants and vitamin C, making them beneficial for overall health. Gongura may help lower blood pressure, improve digestion, and boost the immune system. However, scientific evidence supporting specific medicinal uses is limited, and it should be consumed in moderation as part of a balanced diet.")
                elif predicted_label ==  "Hibiscus Rosa-sinensis":
                    st.subheader("Medicinal Uses")
                    st.write("Hibiscus rosa-sinensis, also known as the Chinese hibiscus or shoe flower, has several traditional medicinal uses. Its flowers and leaves are commonly brewed into tea and are believed to have potential benefits for lowering blood pressure and cholesterol levels, promoting hair growth, and aiding in weight management. However, while it's consumed in traditional remedies, scientific research on these medicinal claims is ongoing, and consultation with a healthcare professional is advisable.")
                elif predicted_label ==  "Jasminum (Jasmine)":
                    st.subheader("Medicinal Uses")
                    st.write("Jasminum, commonly known as Jasmine, is cherished for its aromatic flowers and limited medicinal applications. Jasmine tea, made from its blossoms, is believed to have calming and stress-reducing properties. In traditional medicine, it's used for easing anxiety, promoting sleep, and enhancing skin health. Scientific evidence supporting these claims varies, but Jasmine's pleasing fragrance is often harnessed in aromatherapy for relaxation. Always consult a healthcare professional for specific medicinal recommendations.")
                elif predicted_label ==  "Holy Basil":
                    st.subheader("Medicinal Uses")
                    st.write("Holy Basil (Ocimum sanctum), also known as Tulsi, is revered in Ayurvedic medicine for its various medicinal uses. It is believed to have anti-inflammatory, antioxidant, and adaptogenic properties. Holy Basil is used to alleviate stress, boost the immune system, and improve respiratory health. It's also considered beneficial for digestive issues and as a general tonic. However, consult a healthcare professional before using it for specific medical purposes, as individual responses may vary.")
                elif predicted_label ==  "Indian pennywort":
                    st.subheader("Medicinal Uses")
                    st.write("Indian Pennywort (Centella asiatica), also known as Gotu Kola, has a history of medicinal use. It is believed to enhance cognitive function, improve circulation, and promote wound healing. In traditional medicine, it's used for reducing anxiety, managing skin conditions like psoriasis and eczema, and aiding in the treatment of venous insufficiency. Scientific research supports some of these uses, but consult with a healthcare professional for personalized advice on its medicinal applications.")
                elif predicted_label ==  "Indian Sarsaparilla":
                    st.subheader("Medicinal Uses")
                    st.write(" Indian Sarsaparilla (Hemidesmus indicus) is employed in traditional medicine for its potential medicinal benefits. It is used to treat skin disorders like eczema and psoriasis, alleviate joint pain and inflammation, and support liver health. This herb is also considered a natural diuretic and may help in managing urinary tract issues. However, scientific evidence supporting these uses is limited, and it should be used under professional guidance. ")
                elif predicted_label ==  "Mangifera Indica (Mango)":
                    st.subheader("Medicinal Uses")
                    st.write("Mangifera indica, or the Mango tree, offers various potential medicinal uses. Its leaves are traditionally brewed into a tea for their antioxidant properties, aiding in digestion, and reducing blood pressure. The fruit contains vitamins and minerals, supporting overall health. Additionally, mango kernels are used in some traditional practices for their anti-inflammatory and antimicrobial effects. While mangoes are primarily consumed for their delicious taste, these potential health benefits make them a valuable addition to the diet.")
                elif predicted_label ==  "Indian Stinging Nettle":
                    st.subheader("Medicinal Uses")
                    st.write("Indian Stinging Nettle (Urtica dioica subsp. galeopsifolia) has been utilized in traditional medicine for its medicinal properties. The plant's leaves, despite their stinging hairs, are used for treating conditions like arthritis, allergies, and urinary tract infections. They are believed to have anti-inflammatory and diuretic effects. Indian Stinging Nettle has also been used as a potential remedy for prostate issues. While these traditional uses exist, scientific evidence supporting them is limited, and caution should be exercised when handling the plant. Consulting a healthcare professional is advisable for its safe use.")
                elif predicted_label ==  "Indian Thornapple":
                    st.subheader("Medicinal Uses")
                    st.write("Indian Thornapple (Datura metel), also known as Devil's Trumpet, has a history of traditional medicinal use, although it is highly toxic and should be used with extreme caution. In some traditional practices, it has been used for its potential sedative, analgesic, and anti-inflammatory properties. However, its toxic nature can lead to severe side effects and even fatalities, so it is not recommended for self-medication. Medical supervision is crucial if considering any use of Indian Thornapple for medicinal purposes.")
                elif predicted_label ==  "Indian wormwood":
                    st.subheader("Medicinal Uses")
                    st.write("Indian Wormwood (Artemisia indica), also known as Mugwort, is used in traditional medicine for its potential medicinal benefits. It is believed to have anti-inflammatory, anti-parasitic, and digestive properties. Indian Wormwood has been used to treat digestive disorders, alleviate pain and inflammation, and as a natural remedy for various parasitic infections. However, scientific evidence supporting these uses is limited, and caution should be exercised. Consultation with a healthcare professional is advisable before using Indian Wormwood for medicinal purposes.")
                elif predicted_label ==  "Ivy Gourd":
                    st.subheader("Medicinal Uses")
                    st.write("Ivy Gourd (Coccinia grandis), also known as Kovakkai or Tindora, is used in traditional medicine for its potential health benefits. It is believed to have anti-diabetic properties and may help lower blood sugar levels. Ivy Gourd is also used as a natural remedy for digestive issues like indigestion and constipation. While some studies support its potential in diabetes management, further research is needed to confirm its effectiveness. Consultation with a healthcare professional is advised for proper guidance and safe usage.")
                elif predicted_label ==  "Jasminum (Jasmine)":
                    st.subheader("Medicinal Uses")
                    st.write("Jasminum, commonly known as Jasmine, is primarily valued for its aromatic flowers. In traditional medicine, Jasmine has been used for its potential calming and sedative properties, often used to alleviate anxiety, stress, and insomnia. It is also believed to have mild anti-inflammatory and anti-bacterial effects, occasionally used to treat minor skin irritations. While it is used in aromatherapy and some herbal remedies, scientific evidence supporting these medicinal uses is limited, and professional guidance is advisable.")
                elif predicted_label ==  "Kokilaksha":
                    st.subheader("Medicinal Uses")
                    st.write("Kokilaksha (Asteracantha longifolia) is an herb used in traditional Ayurvedic medicine. It is believed to have diuretic properties and is used for its potential benefits in treating urinary tract infections, kidney stones, and improving overall kidney and urinary system health. Kokilaksha is also considered an aphrodisiac and may be used to enhance sexual vitality. While these traditional uses exist, scientific research on its efficacy is limited, and it should be used with caution under the guidance of a qualified healthcare practitioner.")
                elif predicted_label ==  "Mentha (Mint)":
                    st.subheader("Medicinal Uses")
                    st.write("Mentha, commonly known as Mint, has a long history of medicinal uses. Its leaves are well-known for their digestive properties, helping to alleviate indigestion and nausea. Mint tea or oil can also provide relief from headaches and migraines. Additionally, it has mild analgesic and anti-inflammatory effects, making it useful for minor pain relief. Mint is often used in aromatherapy to promote relaxation and ease respiratory issues. However, it should be used in moderation, as excessive consumption may lead to adverse effects.")
                elif predicted_label ==  "Moringa Oleifera (Drumstick)":
                    st.subheader("Medicinal Uses")
                    st.write("Moringa Oleifera, also known as Drumstick, is a nutrient-rich plant with numerous potential medicinal uses. Its leaves, pods, and seeds are packed with vitamins, minerals, and antioxidants. Moringa is used traditionally to boost the immune system, lower blood sugar levels, reduce inflammation, and promote digestive health. It's also applied topically for wound healing. While scientific research supports some of these benefits, consult with a healthcare professional before using Moringa for specific medicinal purposes, as individual responses may vary.")
                elif predicted_label ==  "Muntingia Calabura (Jamaica Cherry-Gasagase)":
                    st.subheader("Medicinal Uses")
                    st.write("Muntingia calabura, also known as Jamaica Cherry or Gasagase, has limited documented medicinal uses. In some traditional practices, its leaves have been used to make infusions believed to have potential anti-inflammatory properties and to treat ailments like diarrhea and fever. However, scientific evidence supporting these uses is scarce, and caution should be exercised. Consultation with a healthcare professional is advisable before using Muntingia calabura for medicinal purposes, as its safety and efficacy are not well-established.")
                elif predicted_label ==  "Murraya Koenigii (Curry)":
                    st.subheader("Medicinal Uses")
                    st.write("Murraya koenigii, commonly known as Curry Leaves, is a staple in Indian cuisine and also valued for its potential medicinal properties. These aromatic leaves are rich in antioxidants and are traditionally used to aid digestion, lower blood sugar levels, and manage diabetes. They are also believed to have anti-inflammatory and antimicrobial effects, possibly helpful in treating skin conditions and promoting hair health. While curry leaves are primarily used as a culinary ingredient, their medicinal uses align with their dietary inclusion in traditional Indian cooking.")
                elif predicted_label ==  "Nerium Oleander (Oleander)":
                    st.subheader("Medicinal Uses")
                    st.write("Nerium oleander, commonly known as Oleander, contains toxic compounds and should not be used for self-medication. However, some traditional practices have utilized it cautiously in very limited ways. Oleander has been considered in traditional medicine for potential use in the treatment of heart conditions and as an external remedy for skin issues. Its extreme toxicity makes it highly dangerous, and it should never be used without the guidance of a trained healthcare professional, as ingestion or misuse can be fatal.")
                elif predicted_label ==  "Nyctanthes Arbor-tristis (Parijata)":
                    st.subheader("Medicinal Uses")
                    st.write("Nyctanthes arbor-tristis, also known as Parijata or Night-flowering Jasmine, is valued for its medicinal properties in Ayurvedic and traditional medicine. Its leaves and flowers are used to prepare remedies believed to have anti-inflammatory, analgesic, and antipyretic effects. Parijata is traditionally used to treat various ailments, including arthritis, fever, digestive disorders, and skin conditions. While it has a long history of use in traditional systems, scientific research is ongoing to validate its medicinal benefits, and it should be used under professional guidance.")
                elif predicted_label ==  "Ocimum Tenuiflorum (Tulsi)":
                    st.subheader("Medicinal Uses")
                    st.write("Ocimum tenuiflorum, commonly known as Tulsi or Holy Basil, holds a revered place in Ayurvedic medicine. Its leaves are prized for their potential medicinal benefits, including antimicrobial, anti-inflammatory, and adaptogenic properties. Tulsi is used traditionally to boost the immune system, relieve stress, and alleviate respiratory issues. It is also considered beneficial for digestive health and skin conditions. Scientific research supports some of these uses, making Tulsi a widely used herb in natural medicine. However, it's essential to consult with a healthcare professional for specific recommendations and dosages.")
                elif predicted_label ==  "Piper Betle (Betel)":
                    st.subheader("Medicinal Uses")
                    st.write("Piper betle, commonly known as Betel, is a tropical vine known for its leaves, which are traditionally used in various cultural practices. Chewing betel leaves with areca nut and slaked lime is a common tradition in some Asian countries. It is believed to have mild stimulant properties and is often used as a breath freshener. However, the practice is associated with health risks, including oral cancer, due to the carcinogenic nature of areca nut and slaked lime, and is not recommended for medicinal use.")
                elif predicted_label ==  "Plectranthus Amboinicus (Mexican Mint)":
                    st.subheader("Medicinal Uses")
                    st.write("Plectranthus amboinicus, commonly known as Mexican Mint or Indian Borage, is valued in traditional medicine for its potential medicinal properties. Its leaves are used to prepare remedies believed to have anti-inflammatory, antibacterial, and antifungal effects. Mexican Mint is used to alleviate respiratory issues like coughs and asthma, as well as digestive problems. It is also applied topically for wound healing. While these traditional uses exist, scientific research on its efficacy is limited, and professional guidance is advisable before using Mexican Mint for medicinal purposes.")
                elif predicted_label ==  "Pongamia Pinnata (Indian Beech)":
                    st.subheader("Medicinal Uses")
                    st.write("Pongamia pinnata, commonly known as Indian Beech or Karanja, has traditional medicinal uses. The oil extracted from its seeds is used topically for its potential wound-healing, anti-inflammatory, and analgesic properties. It has been applied to treat skin conditions like eczema and psoriasis. Additionally, Indian Beech is used in Ayurvedic medicine for its anti-parasitic effects and as a natural remedy for diabetes. While it has a history of use, more research is needed to confirm its effectiveness, and professional guidance is recommended for medicinal applications.")
                elif predicted_label ==  "Psidium Guajava (Guava)":
                    st.subheader("Medicinal Uses")
                    st.write("Psidium guajava, commonly known as Guava, offers various potential medicinal benefits. Rich in vitamin C and antioxidants, it supports immune health and may help in preventing and managing various illnesses. Guava leaves are used traditionally to treat diarrhea, stomachaches, and respiratory issues. They possess anti-inflammatory properties and can aid in wound healing. While Guava is primarily consumed as a fruit, its leaves and extracts have a place in herbal remedies, but scientific research is ongoing to validate these uses. Consultation with a healthcare professional is advisable for specific medicinal applications.")
                elif predicted_label ==  "Punica Granatum (Pomegranate)":
                    st.subheader("Medicinal Uses")
                    st.write("Punica granatum, commonly known as Pomegranate, is celebrated for its potential medicinal properties. Its antioxidant-rich juice is believed to improve heart health, lower blood pressure, and reduce cholesterol levels. Pomegranate may also have anti-inflammatory effects and support digestion. The fruit is valued for its potential in preventing chronic diseases and promoting overall well-being. While Pomegranate is widely consumed for its health benefits, scientific research continues to explore and validate its many medicinal uses, making it a valuable addition to a balanced diet.")
                elif predicted_label ==  "Purple Tephrosia":
                    st.subheader("Medicinal Uses")
                    st.write("Purple Tephrosia (Tephrosia purpurea) has been used traditionally in Ayurvedic and folk medicine for various medicinal purposes. It is believed to possess anti-inflammatory and analgesic properties, and its roots are used in remedies for joint pain, arthritis, and fever. Additionally, Purple Tephrosia has been employed as a diuretic, for respiratory conditions, and in the treatment of skin disorders. While it has a history of use, scientific research on its efficacy is ongoing, and professional guidance is recommended for its safe and effective use.")
                elif predicted_label ==  "Santalum Album (Sandalwood)":
                    st.subheader("Medicinal Uses")
                    st.write("Santalum album, or Sandalwood, is highly valued for its potential medicinal properties. Its aromatic heartwood is used to extract essential oil, which is utilized in aromatherapy and traditional medicine. Sandalwood oil is believed to have anti-inflammatory, antiseptic, and calming effects. It's used to soothe skin conditions, such as acne and eczema, reduce anxiety and stress, and alleviate respiratory issues. While its traditional uses are well-documented, scientific research continues to explore its medicinal applications, making Sandalwood a sought-after herb for holistic health and well-being.")
                elif predicted_label ==  "Syzygium Cumini (Jamun)":
                    st.subheader("Medicinal Uses")
                    st.write("Syzygium cumini, commonly known as Jamun or Indian Blackberry, has several traditional medicinal uses. Its fruits, leaves, and seeds are used in Ayurvedic and traditional medicine systems. Jamun is believed to have anti-diabetic properties, as it can help regulate blood sugar levels. It's also used to treat digestive disorders, relieve diarrhea, and improve oral health. The leaves and bark are used in various remedies. While Jamun has a history of use, more research is needed to confirm its effectiveness, and it should be used under professional guidance for medicinal purposes.")
                elif predicted_label ==  "Syzygium Jambos (Rose Apple)":
                    st.subheader("Medicinal Uses")
                    st.write("Syzygium jambos, commonly known as Rose Apple, has limited documented medicinal uses. In some traditional practices, its leaves and fruit are used as a remedy for digestive issues, such as diarrhea and indigestion. The leaves are believed to have potential anti-inflammatory properties and have been applied topically for skin problems. However, scientific evidence supporting these uses is scarce, and caution should be exercised. Consulting with a healthcare professional is advisable before using Rose Apple for medicinal purposes.")
                elif predicted_label ==  "Tabernaemontana Divaricata (Crape Jasmine)":
                    st.subheader("Medicinal Uses")
                    st.write("Tabernaemontana divaricata, commonly known as Crape Jasmine, has a limited history of traditional medicinal uses. Its leaves have been used topically for their potential anti-inflammatory properties, and it's occasionally used in Ayurvedic medicine for treating skin conditions like eczema and itching. However, scientific research supporting these uses is limited, and Crape Jasmine is primarily grown for its ornamental value. Caution should be exercised when using it for medicinal purposes, and consultation with a healthcare professional is advisable.")
                elif predicted_label ==  "Trigonella Foenum-graecum (Fenugreek)":
                    st.subheader("Medicinal Uses")
                    st.write("Trigonella foenum-graecum, commonly known as Fenugreek, has a rich history of medicinal uses. Its seeds are prized for their potential health benefits. Fenugreek may help regulate blood sugar levels, reduce cholesterol, and promote digestion. It's also used to increase milk production in breastfeeding mothers and as a traditional remedy for menstrual discomfort. Some studies support these medicinal uses, making Fenugreek a popular choice in herbal medicine. However, individuals with certain medical conditions should use it cautiously and consult with a healthcare professional before incorporating it into their regimen.")
                

if selected == "About":            
# Display some sample images from your dataset
    st.header("What is AyurLeafAI ?")
    st.write("Ayurvedic Plant Species Identification involves the meticulous recognition of various plants used in Ayurvedic medicine based on their unique botanical characteristics. This knowledge is essential for ensuring the safety and effectiveness of herbal remedies. Once a plant is accurately identified, its leaves, among other parts, are often utilized for their medicinal properties.")
    st.write("For instance, in Ayurveda, the identification of Neem leaves (Azadirachta indica) is crucial. Neem leaves are recognized by their pinnate structure with small, serrated leaflets. These leaves are renowned for their powerful antimicrobial and anti-inflammatory properties. They are commonly used in Ayurvedic remedies to treat skin conditions like acne and eczema, as well as to promote overall detoxification and immune system support. Additionally, Neem leaves are used in oral hygiene practices for their antibacterial effects, making them a versatile and highly valued herb in Ayurvedic medicine.")
    st.header("How it works ?")
    st.write("AyurLeafAI, using machine learning, employs a data-driven approach to identify and analyze the unique characteristics of Ayurvedic leaves. It begins by collecting a dataset of leaf images and their corresponding Ayurvedic properties. Machine learning algorithms are then trained on this data to recognize patterns, such as color, shape, and texture, which are indicative of medicinal qualities. When a user submits a leaf image, the system processes it through these algorithms, comparing it to the learned patterns. It then predicts the potential medicinal properties, helping users identify the therapeutic benefits of the leaf based on Ayurvedic principles, promoting natural and holistic healthcare choices.")
    image = Image.open('architecture.png')
    st.image(image, caption='Fig : System Architecture of AyurLeafAI')
    st.subheader("Sample plant images")
    class_folders = os.listdir(main_data_dir)
    num_samples = min(len(class_folders), 5)  # Show up to 5 samples
    images_per_row = 5

    for i in range(num_samples):
        class_folder = class_folders[i]
        class_folder_path = os.path.join(main_data_dir, class_folder)
        image_files = [f for f in os.listdir(class_folder_path) if f.endswith('.jpg')]

        if image_files:
            st.subheader(class_folder)
            fig, axs = plt.subplots(1, images_per_row, figsize=(15, 15))

            for j in range(images_per_row):
                image_path = os.path.join(class_folder_path, image_files[j])
                img = mpimg.imread(image_path)
                axs[j].imshow(img)
                axs[j].set_title(f"Sample {j + 1}")
                axs[j].axis('off')

            st.pyplot(fig)

# Add any additional content or information about your model or dataset here.
