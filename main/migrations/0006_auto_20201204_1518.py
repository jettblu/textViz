# Generated by Django 3.1.2 on 2020-12-04 23:18

from django.db import migrations
import picklefield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0005_auto_20201204_1511'),
    ]

    operations = [
        migrations.AddField(
            model_name='messagedocument',
            name='contacts',
            field=picklefield.fields.PickledObjectField(default=1, editable=False),
        ),
        migrations.DeleteModel(
            name='Contacts',
        ),
    ]
