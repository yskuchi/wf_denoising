
void read_wf_macro(TString filename)
{
   //% ./meganalyzer -I 'read_wf_macro.C("raw11100.root")'


   Int_t kNPoints = 512; // number of points of a wf to be output
   Bool_t kReadCDCH = true;
   Bool_t kReadSPX = false;
   Bool_t kOnly1stClusterTiming = true;
   
   //Int_t maxEvent = -1;// 500;//-1; // MC
   Int_t maxEvent = 500;//-1;   // 2018

   const Double_t kSignalVelocity = 2.98e10;

   TChain *raw = new TChain("raw");
   raw->Add(filename);
   auto nentry = raw->GetEntries();
   cout << "Number of events = " << nentry << endl;

   TTreeReader reader(raw);
   TTreeReaderArray<MEGDRSChip>   chipRA(reader, "drschip");

   // rec file to get CYLDCHWireRunHeader
   TString recfilename = filename;
   recfilename.ReplaceAll("raw", "rec");
   TString simfilename = filename;
   simfilename.ReplaceAll("raw", "sim");

   TChain *sim = nullptr;


   vector<Int_t> addresses; // reorder addresses by wire IDs.
   vector<Int_t> channels;

   if (TFile *recfile = TFile::Open(recfilename); recfile) {
      if (kReadCDCH) {
         auto pWireRunHeaders = (TClonesArray*)(recfile->Get("CYLDCHWireRunHeader"));
         for (Int_t iWire = 0; iWire < pWireRunHeaders->GetSize(); iWire++) {
            auto pWireRunHeader = static_cast<MEGCYLDCHWireRunHeader*>(pWireRunHeaders->At(iWire));
            if (!pWireRunHeader->GetActive()) {
               continue;
            }
            if (pWireRunHeader->GetDRSAddress_u() < 0 || pWireRunHeader->GetDRSAddress_d() < 0) {
               continue;
            }
            addresses.push_back(pWireRunHeader->GetDRSAddress_u());
            addresses.push_back(pWireRunHeader->GetDRSAddress_d());
            channels.push_back(iWire * 2);
            channels.push_back(iWire * 2 + 1);
         }
      }
      if (kReadSPX) {
         auto pPPDRunHeaders = (TClonesArray*)(recfile->Get("SPXPPDRunHeader"));
         for (Int_t iPPD = 0; iPPD < pPPDRunHeaders->GetSize(); iPPD++) {
            auto pPPDRunHeader = static_cast<MEGSPXPPDRunHeader*>(pPPDRunHeaders->At(iPPD));
            if (pPPDRunHeader->GetDRSAddress() < 0) {
               continue;
            }
            addresses.push_back(pPPDRunHeader->GetDRSAddress());
            channels.push_back(iPPD);
         }
      }
      recfile->Close();
      delete recfile;
   } else if (TFile *simfile = TFile::Open(simfilename); simfile) {
      if (kReadCDCH) {

         sim = new TChain("sim");
         sim->Add(simfilename);

         auto pWireRunHeaders = (TClonesArray*)(simfile->Get("BarCYLDCHWireRunHeader"));
         for (Int_t iWire = 0; iWire < pWireRunHeaders->GetSize(); iWire++) {
            auto pWireRunHeader = static_cast<MEGBarCYLDCHWireRunHeader*>(pWireRunHeaders->At(iWire));
            //if (!pWireRunHeader->GetActive()) continue;
            if (pWireRunHeader->GetDRSAddress_u() < 0 || pWireRunHeader->GetDRSAddress_d() < 0) {
               continue;
            }
            addresses.push_back(pWireRunHeader->GetDRSAddress_u());
            addresses.push_back(pWireRunHeader->GetDRSAddress_d());
            channels.push_back(iWire * 2);
            channels.push_back(iWire * 2 + 1);
         }
      }
      if (kReadSPX) {
         auto pPPDRunHeaders = (TClonesArray*)(simfile->Get("BarSPXPPDRunHeader"));
         for (Int_t iPPD = 0; iPPD < pPPDRunHeaders->GetSize(); iPPD++) {
            auto pPPDRunHeader = static_cast<MEGBarSPXPPDRunHeader*>(pPPDRunHeaders->At(iPPD));
            if (pPPDRunHeader->GetDRSAddress() < 0) {
               continue;
            }
            addresses.push_back(pPPDRunHeader->GetDRSAddress());
            channels.push_back(iPPD);
         }
      }
      simfile->Close();
      delete simfile;
   }
   auto addressMin = std::min_element(addresses.begin(), addresses.end());
   auto addressMax = std::max_element(addresses.begin(), addresses.end());
   Int_t addressSelect[] = {*addressMin, *addressMax}; // 2021 or new MC
   cout << "Address " << *addressMin << " - " << *addressMax << endl;


   TString csvfile = filename;
   csvfile.ReplaceAll(".root", ".csv");
   csvfile.ReplaceAll("raw", "wf");
   if (kReadCDCH) {
      csvfile.ReplaceAll("wf", "wf_cdch");
   }
   if (kReadSPX) {
      csvfile.ReplaceAll("wf", "wf_spx");
   }
   csvfile = basename(csvfile.Data());
   ofstream fout(csvfile.Data(), std::ios::out);

   TString csvfile2 = filename;
   csvfile2.ReplaceAll(".root", ".csv");
   csvfile2.ReplaceAll("raw", "cls");
   if (kReadCDCH) {
      if (kOnly1stClusterTiming) {
         csvfile2.ReplaceAll("cls", "cls1st_cdch");
      } else {
         csvfile2.ReplaceAll("cls", "cls_cdch");
      }
   }
   if (kReadSPX) {
      csvfile2.ReplaceAll("cls", "cls_spx");
   }
   csvfile2 = basename(csvfile2.Data());
   ofstream fout2(csvfile2.Data(), std::ios::out);

   std::vector<Float_t> drs(1024);
   MEGDRSWaveform wf[8];
   Int_t iEvent(-1);
   Int_t nTotData(0);
   Int_t nChPrev(0);

   TTreeReader simReader(sim);
   TTreeReaderValue<MEGMCCYLDCHEvent>      mceventRV(simReader, "mccyldch.");

   cout << "Start event loop for " << nentry << " events" << endl;
   while (reader.Next()) {

      if (sim) {
         simReader.Next();
      }

      ++iEvent;
      if (iEvent % 100 == 1) {
         cout << iEvent << " events finished." << endl;
      }

      if (maxEvent > 0 && iEvent > maxEvent) {
         break;
      }

      Double_t binSize(0), timeMin(0);
      map<Int_t, vector<Float_t> > wfs;
      for (auto&& chip : chipRA) {
         auto chipData = chip.GetDRSChipData();
         auto address = chipData->GetAddress();
         if (address < addressSelect[0] || address > addressSelect[1]) {
            continue;
         }
         if (!chipData->HaveData()) {
            continue;
         }
         for (Int_t iChannel = 0; iChannel < 8; iChannel++) {
            chipData->SetWaveformAt(iChannel, &wf[iChannel]);
         }
         chipData->DecodeWaves();
         for (Int_t iChannel = 0; iChannel < 8; iChannel++) {
            if (!wf[iChannel].GetNPoints()) {
               continue;
            }
            Int_t addressCh = address + iChannel;
            auto ampl = wf[iChannel].GetAmplitude();
            wfs[addressCh].resize(wf[iChannel].GetNPoints(), 0);
            std::copy(ampl, ampl + wf[iChannel].GetNPoints(), wfs[addressCh].begin());
            binSize = wf[iChannel].GetBinSize();
            timeMin = wf[iChannel].GetTimeMin();
         }
      }

      map<Int_t, vector<Float_t> > clusters;
      if (sim) {
         Int_t nMCHits = mceventRV->GetCYLDCHMCHitSize();
         map<Int_t, set<Int_t> > wireHitMap;
         for (Int_t iHit = 0; iHit < nMCHits; iHit++) {
            // Check all the MCHits and group them into wires
            auto mchit = mceventRV->GetCYLDCHMCHitAt(iHit);
            Int_t wire = mchit->Getwire();
            wireHitMap[wire].insert(iHit);
         }

         MEGWaveform *mcwfsum[2];
         for (Int_t iside = 0; iside < 2; iside++) {
            mcwfsum[iside] = new MEGWaveform(kNPoints, binSize, timeMin, "");
         }
         vector<MEGWaveform*> mcwfs;
         for (size_t iCh = 0; iCh < addresses.size(); iCh += 2) {
            if (wfs.find(addresses[iCh]) != wfs.end()
                && wfs.find(addresses[iCh + 1]) != wfs.end()) {
               Int_t iWire = channels[iCh] / 2;
               if (wireHitMap.find(iWire) == wireHitMap.end()) {
                  continue;
               }

               for (Int_t iside = 0; iside < 2; iside++) {
                  mcwfsum[iside]->ResetAmplitude();
               }

               Int_t iHit(-1);
               for (auto &&hitindex : wireHitMap[iWire]) {
                  auto mchit = mceventRV->GetCYLDCHMCHitAt(hitindex);
                  ++iHit;

                  MEGWaveform *mcwf[2];
                  if ((iHit + 1) * 2 < mcwfs.size()) {
                     for (Int_t iside = 0; iside < 2; iside++) {
                        mcwf[iside] = mcwfs[iHit * 2 + iside];
                        mcwf[iside]->ResetAmplitude();
                     }
                  } else {
                     for (Int_t iside = 0; iside < 2; iside++) {
                        mcwf[iside] = new MEGWaveform(kNPoints, binSize, timeMin, "");
                        mcwfs.push_back(mcwf[iside]);
                     }
                  }
                  Int_t nclusters = mchit->Getnclusters();
                  for (Int_t iCluster = 0; iCluster < nclusters; iCluster++) {
                     for (Int_t iside = 0; iside < 2; iside++) {
                        Double_t clsSize(0.), clsLn(0.);
                        if (iside == 0) {
                           clsSize = mchit->GetclsSizeUSAt(iCluster);
                           clsLn = mchit->GetclsLnUSAt(iCluster);
                        } else {
                           clsSize = mchit->GetclsSizeDSAt(iCluster);
                           clsLn = mchit->GetclsLnDSAt(iCluster);
                        }
                        Double_t clsTime = mchit->GetclsTimeAt(iCluster);
                        clsTime += clsLn / kSignalVelocity;
                        clsSize *= 0.01 / 500000;// scale adjustment 10mV <-> 5e5
                        Int_t bin = mcwf[iside]->FindPoint(clsTime);
                        if (bin < 0 || bin >= mcwf[iside]->GetNPoints() - 1) {
                           // out of DRS window
                        } else {
                           if (clsTime - mcwf[iside]->GetTimeAt(bin)
                               > mcwf[iside]->GetTimeAt(bin + 1) - clsTime) {
                              // closer to the next point
                              bin++;
                           }
                           if (kOnly1stClusterTiming) {
                              if (iCluster == 0) {
                                 mcwf[iside]->SetAmplitudeAt(bin, 1);
                                 mcwfsum[iside]->SetAmplitudeAt(bin, 1);
                              }
                           } else {
                              mcwfsum[iside]->AddAmplitudeAt(bin, clsSize);
                           }
                        }
                     }
                  }
               }
               for (Int_t iside = 0; iside < 2; iside++) {
                  auto ampl = mcwfsum[iside]->GetAmplitude();
                  clusters[addresses[iCh + iside]].resize(mcwfsum[iside]->GetNPoints(), 0);
                  std::copy(ampl, ampl + mcwfsum[iside]->GetNPoints(), clusters[addresses[iCh + iside]].begin());
               }
            }
         }

         delete mcwfsum[0];
         delete mcwfsum[1];
         for (auto &&wf : mcwfs) {
            delete wf;
         }
         mcwfs.clear();
      }
      Int_t nCh(0);
      for (size_t iCh = 0; iCh < addresses.size(); iCh += 2) {
         if (wfs.find(addresses[iCh]) != wfs.end()
             && wfs.find(addresses[iCh + 1]) != wfs.end()) {
            string line;
            for (Int_t iEnd = 0; iEnd < 2; iEnd++) {
               drs.clear();
               drs.resize(1024, 0);
               std::copy(wfs[addresses[iCh + iEnd]].begin(),
                         wfs[addresses[iCh + iEnd]].end(), drs.begin());
               drs.resize(kNPoints);

               for_each(drs.begin(), drs.end(), [&](Double_t a) {
                  line += std::to_string(a) + ",";
               });
               line.pop_back();
               line += '\n';
               ++nCh;
               ++nTotData;
            }
            fout << line;

            line.clear();
            if (sim) {
               for (Int_t iEnd = 0; iEnd < 2; iEnd++) {
                  drs.clear();
                  drs.resize(1024, 0);
                  std::copy(clusters[addresses[iCh + iEnd]].begin(),
                            clusters[addresses[iCh + iEnd]].end(), drs.begin());
                  drs.resize(kNPoints);
                  for_each(drs.begin(), drs.end(), [&](Double_t a) {
                     line += std::to_string(a) + ",";
                  });
                  line.pop_back();
                  line += '\n';
               }
               fout2 << line;
            }
         }
      }
      if (nCh != nChPrev) {
         cout << "Number of active channels " << nCh << endl;
         nChPrev = nCh;
      }
   }
   cout << "Total number of samples: " << nTotData << endl;
   fout.close();
   fout2.close();

}
