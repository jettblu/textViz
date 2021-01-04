from django.db import models
from django.contrib.auth.models import User
from datetime import datetime
from picklefield.fields import PickledObjectField

# Create your models here.


class MessageDocument(models.Model):
    msgFile = models.FileField()

    uploadDate = models.DateTimeField(auto_now_add=True)
    # attaches file to a specific user
    documentOwner = models.ForeignKey(User, default=1, verbose_name='User', on_delete=models.SET_DEFAULT,
                                      related_name='files')
    contacts = PickledObjectField(default=1)

    def __str__(self):
        return self.msgFile.name
