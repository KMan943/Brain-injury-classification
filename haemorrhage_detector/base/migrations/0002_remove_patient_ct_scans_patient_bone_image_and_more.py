# Generated by Django 5.1.2 on 2024-11-06 19:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='patient',
            name='Ct_scans',
        ),
        migrations.AddField(
            model_name='patient',
            name='bone_image',
            field=models.ImageField(default='', upload_to='patients/bone/'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='patient',
            name='brain_image',
            field=models.ImageField(default='', upload_to='patients/brain/'),
            preserve_default=False,
        ),
    ]