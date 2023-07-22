package app.ij.mlwithtensorflowlite;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import com.github.barteksc.pdfviewer.PDFView;

import java.io.IOException;
import java.io.InputStream;

public class blog_activity extends AppCompatActivity {

    TextView tv_test;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_blog2);

        PDFView pdfView = findViewById(R.id.pdf_view); // Replace with your PDFView id

        try {
            InputStream is = getAssets().open("my_pdf_file.pdf"); // Replace with your PDF file name
            pdfView.fromStream(is).load();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}