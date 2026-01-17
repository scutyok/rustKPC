Name "nightmarepatrol1"

AbortOnDamage FALSE
AbortOnEnemySighted FALSE
CanBeMoved	FALSE
SightRange 512
TimeOut 999.0
FaceMoveDir TRUE
YawRate 180
WalkSpeed 100
CheckForEnemyInterval 0.5
TargetPlayer

Loop1:

OnEnemyOutOfRange_Goto DISABLE			
OnEnemySighted_Goto TurretAttack				
OnDamage_Goto FarAttack			
TimeOut 10.0


StartAnim "Run"
TurnToPos 1136 320 -672
MoveToPos 1136 320 -672
TurnToPos 1136 320 -256
StartAnim "AmbB"
Wait 1.0
StartAnim "Run"
TurnToPos 1376 320 -672
MoveToPos 1376 320 -672
TurnToPos 1376 320 -256
StartAnim "AmbB"
Wait 1.0

GoTo Loop1

FarAttack:
	If_DamageFromPlayer_Goto	BoostSightRange

//	Goto		Loop1
       
BoostSightRange:
	SightRange	1600.0

TurretAttack:

	OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	        Loop1
	
	YawRate		360.0

	TurnToTarget	0.8	
	RangedAttackTarget
        MeleeAttackTarget


        Goto	               TurretAttack